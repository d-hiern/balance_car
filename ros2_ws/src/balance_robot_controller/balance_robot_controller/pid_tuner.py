"""
PID Auto-Tuner Node - Tự động tìm hệ số PID tối ưu cho xe cân bằng.

Phương pháp: Relay Feedback (Åström-Hägglund) kết hợp Ziegler-Nichols.

Quy trình tự động:
  Phase 1 - STABILIZE: Ổn định robot bằng PD controller cơ bản
  Phase 2 - RELAY:     Áp dụng relay feedback để tạo dao động ổn định
  Phase 3 - ANALYZE:   Phân tích dao động → tính Ku (ultimate gain) và Tu (period)
  Phase 4 - COMPUTE:   Tính Kp, Ki, Kd theo quy tắc Ziegler-Nichols
  Phase 5 - VERIFY:    Kiểm nghiệm bộ PID đã tính trên robot thực
  Phase 6 - DONE:      Xuất kết quả + lưu file YAML

Subscribes:
    /imu/data (sensor_msgs/Imu)

Publishes:
    /cmd_vel (geometry_msgs/Twist)
    /pid_tuner/status (std_msgs/String)
    /pid_tuner/data (std_msgs/Float64MultiArray)

Sử dụng:
    ros2 launch balance_robot_controller tuning.launch.py

Hoặc chạy trực tiếp:
    ros2 run balance_robot_controller pid_tuner
"""

import math
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, String

from balance_robot_controller.pid import PIDController


def quaternion_to_pitch(q):
    """Trích xuất góc pitch từ quaternion."""
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    if abs(sinp) >= 1.0:
        return math.copysign(math.pi / 2.0, sinp)
    return math.asin(sinp)


class PIDTunerNode(Node):
    """
    Node tự động điều chỉnh PID cho xe cân bằng hai bánh.

    Sử dụng phương pháp Relay Feedback kết hợp với bộ điều khiển
    ổn định cơ bản (vì xe cân bằng là hệ thống mất ổn định).

    Thuật toán:
    1. Ổn định robot bằng PD controller đơn giản
    2. Thêm relay (bang-bang) lên trên: u = Kp_stab*e + Kd_stab*(-ω) + d*sign(e)
    3. Đo biên độ dao động (a) và chu kỳ (Tu) khi đạt trạng thái ổn định
    4. Tính ultimate gain: Ku = 4*d / (π*a)
    5. Áp dụng công thức Ziegler-Nichols để tính Kp, Ki, Kd
    """

    # === Các phase tuning ===
    PHASE_WAIT = 'WAIT'
    PHASE_STABILIZE = 'STABILIZE'
    PHASE_RELAY = 'RELAY'
    PHASE_ANALYZE = 'ANALYZE'
    PHASE_VERIFY = 'VERIFY'
    PHASE_DONE = 'DONE'

    # === Quy tắc Ziegler-Nichols ===
    ZN_RULES = {
        'classic': {
            'name': 'ZN Classic PID',
            'kp_factor': 0.60,
            'ti_factor': 0.50,
            'td_factor': 0.125
        },
        'some_overshoot': {
            'name': 'ZN Some Overshoot',
            'kp_factor': 0.33,
            'ti_factor': 0.50,
            'td_factor': 0.33
        },
        'no_overshoot': {
            'name': 'ZN No Overshoot',
            'kp_factor': 0.20,
            'ti_factor': 0.50,
            'td_factor': 0.33
        },
        'pessen': {
            'name': 'Pessen Integration',
            'kp_factor': 0.70,
            'ti_factor': 0.40,
            'td_factor': 0.15
        },
    }

    def __init__(self):
        super().__init__('pid_tuner')

        # ===== Parameters =====
        self.declare_parameter('relay_amplitude', 0.3)
        self.declare_parameter('stabilizing_kp', 30.0)
        self.declare_parameter('stabilizing_kd', 3.0)
        self.declare_parameter('num_cycles', 6)
        self.declare_parameter('zn_rule', 'some_overshoot')
        self.declare_parameter('stabilize_duration', 3.0)
        self.declare_parameter('verify', True)
        self.declare_parameter('verify_duration', 8.0)
        self.declare_parameter('fall_threshold', 0.785)
        self.declare_parameter('max_velocity', 1.5)
        self.declare_parameter('output_file', '~/tuned_pid_params.yaml')

        # Lấy giá trị
        self.relay_amplitude = self.get_parameter('relay_amplitude').value
        self.stab_kp = self.get_parameter('stabilizing_kp').value
        self.stab_kd = self.get_parameter('stabilizing_kd').value
        self.num_cycles = self.get_parameter('num_cycles').value
        self.zn_rule = self.get_parameter('zn_rule').value
        self.stabilize_duration = self.get_parameter('stabilize_duration').value
        self.do_verify = self.get_parameter('verify').value
        self.verify_duration = self.get_parameter('verify_duration').value
        self.fall_threshold = self.get_parameter('fall_threshold').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.output_file = self.get_parameter('output_file').value

        # ===== Trạng thái tuner =====
        self.phase = self.PHASE_WAIT
        self.phase_start_time = None
        self.start_time = None

        # Dữ liệu relay
        self.relay_data = []       # [(timestamp, pitch, output)]
        self.zero_crossings = []   # [timestamp] - thời điểm pitch đi qua 0
        self.peaks = []            # [amplitude] - biên độ đỉnh
        self.prev_sign = 0
        self.cycle_count = 0
        self.half_cycle_max_pitch = 0.0  # Đỉnh pitch trong nửa chu kỳ hiện tại

        # Kết quả tuning
        self.ku = 0.0   # Ultimate gain
        self.tu = 0.0   # Ultimate period
        self.computed_kp = 0.0
        self.computed_ki = 0.0
        self.computed_kd = 0.0

        # Verification
        self.verify_pid = None
        self.verify_errors = []

        # ===== Publishers =====
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, 'pid_tuner/status', 10)
        self.data_pub = self.create_publisher(
            Float64MultiArray, 'pid_tuner/data', 10
        )

        # ===== Subscriber =====
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10
        )

        # ===== Log khởi động =====
        self.get_logger().info('')
        self.get_logger().info('=' * 55)
        self.get_logger().info('  PID AUTO-TUNER - Tự động điều chỉnh hệ số PID')
        self.get_logger().info('=' * 55)
        self.get_logger().info(f'  Phương pháp:       Relay Feedback + Ziegler-Nichols')
        self.get_logger().info(f'  Relay amplitude:   {self.relay_amplitude}')
        self.get_logger().info(f'  Stabilizing Kp:    {self.stab_kp}')
        self.get_logger().info(f'  Stabilizing Kd:    {self.stab_kd}')
        self.get_logger().info(f'  Số chu kỳ cần đo:  {self.num_cycles}')
        self.get_logger().info(f'  ZN Rule:           {self.zn_rule}')
        self.get_logger().info(f'  Verification:      {self.do_verify}')
        self.get_logger().info('=' * 55)
        self.get_logger().info('Đang chờ dữ liệu IMU...')
        self.get_logger().info('')

    # ================================================================
    #                      UTILITY METHODS
    # ================================================================

    def publish_status(self, text):
        """Publish trạng thái tuner."""
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def publish_data(self, pitch, error, output, phase_id):
        """
        Publish dữ liệu tuning cho monitoring.
        Data: [pitch, error, output, phase_id]
        """
        msg = Float64MultiArray()
        msg.layout.dim = [MultiArrayDimension(
            label='tuner_data', size=4, stride=4
        )]
        msg.data = [pitch, error, output, float(phase_id)]
        self.data_pub.publish(msg)

    def publish_cmd_vel(self, linear_x):
        """Publish lệnh vận tốc (clamped)."""
        twist = Twist()
        twist.linear.x = max(-self.max_velocity,
                             min(self.max_velocity, linear_x))
        self.cmd_vel_pub.publish(twist)

    # ================================================================
    #                     MAIN IMU CALLBACK
    # ================================================================

    def imu_callback(self, msg):
        """State machine chính - xử lý theo phase hiện tại."""
        pitch = quaternion_to_pitch(msg.orientation)
        gyro_y = msg.angular_velocity.y
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.start_time is None:
            self.start_time = timestamp

        # Phát hiện ngã
        if abs(pitch) > self.fall_threshold:
            if self.phase not in [self.PHASE_DONE, self.PHASE_WAIT]:
                self.get_logger().error(
                    f'Robot NGÃ! Pitch = {math.degrees(pitch):.1f}°. '
                    f'Dừng tuning.'
                )
                self.publish_cmd_vel(0.0)
                self.phase = self.PHASE_DONE
                self.publish_status('FAILED - Robot fell')
                return

        # State machine
        if self.phase == self.PHASE_WAIT:
            self._enter_stabilize(timestamp)

        elif self.phase == self.PHASE_STABILIZE:
            self._do_stabilize(pitch, gyro_y, timestamp)

        elif self.phase == self.PHASE_RELAY:
            self._do_relay(pitch, gyro_y, timestamp)

        elif self.phase == self.PHASE_ANALYZE:
            self._do_analyze()

        elif self.phase == self.PHASE_VERIFY:
            self._do_verify(pitch, gyro_y, timestamp)

        elif self.phase == self.PHASE_DONE:
            self.publish_cmd_vel(0.0)

    # ================================================================
    #                     PHASE: STABILIZE
    # ================================================================

    def _enter_stabilize(self, timestamp):
        """Chuyển sang phase STABILIZE."""
        self.phase = self.PHASE_STABILIZE
        self.phase_start_time = timestamp
        self.get_logger().info('')
        self.get_logger().info('━' * 50)
        self.get_logger().info('  Phase 1/5: ỔN ĐỊNH ROBOT')
        self.get_logger().info(f'  Sử dụng PD: Kp={self.stab_kp}, Kd={self.stab_kd}')
        self.get_logger().info(f'  Thời gian chờ: {self.stabilize_duration}s')
        self.get_logger().info('━' * 50)
        self.publish_status('STABILIZE')

    def _do_stabilize(self, pitch, gyro_y, timestamp):
        """Phase STABILIZE: Ổn định robot trước khi tuning."""
        elapsed = timestamp - self.phase_start_time

        # PD control đơn giản để ổn định (pitch > 0 => output > 0 để tiến đón trọng tâm)
        error = pitch - 0.0
        output = self.stab_kp * error + self.stab_kd * gyro_y
        self.publish_cmd_vel(output)
        self.publish_data(pitch, error, output, 1.0)

        if elapsed > self.stabilize_duration:
            if abs(pitch) < 0.1:  # < ~6 degrees
                self.get_logger().info(
                    f'  ✓ Robot ổn định! (pitch = {math.degrees(pitch):.2f}°)'
                )
                self._enter_relay(timestamp)
            else:
                self.get_logger().warn(
                    f'  Chưa ổn định (pitch = {math.degrees(pitch):.2f}°). '
                    f'Chờ thêm...'
                )
                self.phase_start_time = timestamp

    # ================================================================
    #                      PHASE: RELAY
    # ================================================================

    def _enter_relay(self, timestamp):
        """Chuyển sang phase RELAY."""
        self.phase = self.PHASE_RELAY
        self.phase_start_time = timestamp
        self.relay_data = []
        self.zero_crossings = []
        self.peaks = []
        self.cycle_count = 0
        self.prev_sign = 0
        self.half_cycle_max_pitch = 0.0

        self.get_logger().info('')
        self.get_logger().info('━' * 50)
        self.get_logger().info('  Phase 2/5: RELAY FEEDBACK')
        self.get_logger().info(f'  Relay amplitude: {self.relay_amplitude}')
        self.get_logger().info(f'  Cần đo: {self.num_cycles} chu kỳ dao động')
        self.get_logger().info('  Đang tạo dao động...')
        self.get_logger().info('━' * 50)
        self.publish_status('RELAY')

    def _do_relay(self, pitch, gyro_y, timestamp):
        """
        Phase RELAY: Áp dụng relay + stabilizing control.

        Output = Kp_stab * error + Kd_stab * (-gyro) + d * sign(error)

        Relay tạo dao động ổn định. Ta đo biên độ và chu kỳ.
        """
        error = pitch - 0.0

        # Stabilizing base control (giữ robot không ngã)
        stab_output = self.stab_kp * error + self.stab_kd * gyro_y

        # Relay component (tạo dao động)
        if error > 0:
            relay_output = self.relay_amplitude
        else:
            relay_output = -self.relay_amplitude

        # Tổng hợp
        output = stab_output + relay_output
        self.publish_cmd_vel(output)
        self.publish_data(pitch, error, output, 2.0)

        # Lưu dữ liệu
        self.relay_data.append((timestamp, pitch, output))

        # Theo dõi đỉnh pitch trong nửa chu kỳ hiện tại
        self.half_cycle_max_pitch = max(
            self.half_cycle_max_pitch, abs(pitch)
        )

        # Phát hiện zero-crossing (pitch đi qua 0)
        current_sign = 1 if pitch >= 0 else -1

        if self.prev_sign == 0:
            self.prev_sign = current_sign
            return

        if current_sign != self.prev_sign:
            self.zero_crossings.append(timestamp)
            self.cycle_count += 1

            # Lưu đỉnh của nửa chu kỳ vừa qua
            if self.half_cycle_max_pitch > 0.0001:
                self.peaks.append(self.half_cycle_max_pitch)

            self.half_cycle_max_pitch = 0.0  # Reset cho nửa chu kỳ mới

            self.get_logger().info(
                f'  Zero-crossing #{self.cycle_count:>2d} | '
                f'pitch = {math.degrees(pitch):>+7.3f}° | '
                f'peak = {math.degrees(self.peaks[-1]) if self.peaks else 0:.3f}°'
            )

        self.prev_sign = current_sign

        # Kiểm tra đủ chu kỳ chưa (mỗi chu kỳ = 2 zero-crossings)
        if self.cycle_count >= self.num_cycles * 2:
            self.get_logger().info(
                f'  ✓ Đã đo đủ {self.num_cycles} chu kỳ!'
            )
            self.phase = self.PHASE_ANALYZE

    # ================================================================
    #                     PHASE: ANALYZE
    # ================================================================

    def _do_analyze(self):
        """Phase ANALYZE: Phân tích dữ liệu relay để tính Ku và Tu."""
        self.get_logger().info('')
        self.get_logger().info('━' * 50)
        self.get_logger().info('  Phase 3/5: PHÂN TÍCH DỮ LIỆU')
        self.get_logger().info('━' * 50)
        self.publish_status('ANALYZE')

        # --- Tính chu kỳ dao động Tu ---
        if len(self.zero_crossings) < 4:
            self.get_logger().error(
                '  ✗ Không đủ zero-crossings! Cần ít nhất 4.'
            )
            self.phase = self.PHASE_DONE
            self.publish_status('FAILED - Not enough data')
            return

        # Bỏ 2 crossings đầu (giai đoạn quá độ)
        crossings = self.zero_crossings[2:]

        # Tính nửa chu kỳ (khoảng cách giữa các zero-crossings liên tiếp)
        half_periods = []
        for i in range(1, len(crossings)):
            hp = crossings[i] - crossings[i - 1]
            if hp > 0.001:  # Lọc bỏ giá trị quá nhỏ
                half_periods.append(hp)

        if not half_periods:
            self.get_logger().error('  ✗ Không thể tính chu kỳ dao động!')
            self.phase = self.PHASE_DONE
            return

        avg_half_period = sum(half_periods) / len(half_periods)
        self.tu = 2.0 * avg_half_period  # Chu kỳ đầy đủ

        # --- Tính biên độ dao động ---
        if len(self.peaks) < 3:
            self.get_logger().error(
                '  ✗ Không đủ peaks! Cần ít nhất 3.'
            )
            self.phase = self.PHASE_DONE
            return

        # Bỏ peak đầu (quá độ), lấy trung bình
        valid_peaks = self.peaks[1:]
        avg_amplitude = sum(valid_peaks) / len(valid_peaks)

        if avg_amplitude < 0.0005:  # < 0.03 degrees
            self.get_logger().error('  ✗ Biên độ dao động quá nhỏ!')
            self.phase = self.PHASE_DONE
            return

        # --- Tính Ultimate Gain Ku ---
        # Ku = 4*d / (π*a)
        # d = relay_amplitude, a = biên độ dao động
        self.ku = 4.0 * self.relay_amplitude / (math.pi * avg_amplitude)

        self.get_logger().info(
            f'  Biên độ dao động (a): {math.degrees(avg_amplitude):.4f}°'
        )
        self.get_logger().info(
            f'  Chu kỳ dao động (Tu): {self.tu:.4f} s '
            f'({1.0/self.tu:.2f} Hz)'
        )
        self.get_logger().info(f'  Ultimate Gain  (Ku): {self.ku:.4f}')
        self.get_logger().info(f'  Ultimate Period(Tu): {self.tu:.4f} s')

        # --- Tính hệ số PID ---
        self._compute_pid_params()

        # Chuyển sang verify hoặc done
        if self.do_verify:
            self._enter_verify()
        else:
            self._print_final_results()
            self.phase = self.PHASE_DONE
            self.publish_status('DONE')

    def _compute_pid_params(self):
        """Tính hệ số PID từ Ku, Tu theo quy tắc Ziegler-Nichols."""
        self.get_logger().info('')
        self.get_logger().info('━' * 50)
        self.get_logger().info('  Phase 4/5: TÍNH HỆ SỐ PID')
        self.get_logger().info('━' * 50)

        # Tính cho quy tắc đã chọn
        rule = self.ZN_RULES.get(
            self.zn_rule, self.ZN_RULES['some_overshoot']
        )

        self.computed_kp = rule['kp_factor'] * self.ku
        ti = rule['ti_factor'] * self.tu  # Integral time
        td = rule['td_factor'] * self.tu  # Derivative time
        self.computed_ki = self.computed_kp / ti if ti > 0 else 0.0
        self.computed_kd = self.computed_kp * td

        self.get_logger().info(f'  Quy tắc: {rule["name"]}')
        self.get_logger().info(f'  >> Kp = {self.computed_kp:.4f}')
        self.get_logger().info(f'  >> Ki = {self.computed_ki:.4f}')
        self.get_logger().info(f'  >> Kd = {self.computed_kd:.4f}')
        self.get_logger().info(f'  (Ti = {ti:.4f}s, Td = {td:.4f}s)')

        # So sánh tất cả các quy tắc
        self.get_logger().info('')
        self.get_logger().info('  --- So sánh tất cả quy tắc ZN ---')
        for rule_name, rule_params in self.ZN_RULES.items():
            kp = rule_params['kp_factor'] * self.ku
            ti_r = rule_params['ti_factor'] * self.tu
            td_r = rule_params['td_factor'] * self.tu
            ki = kp / ti_r if ti_r > 0 else 0.0
            kd = kp * td_r
            marker = ' ◄' if rule_name == self.zn_rule else ''
            self.get_logger().info(
                f'  {rule_params["name"]:>25s}: '
                f'Kp={kp:>8.3f}  Ki={ki:>8.3f}  Kd={kd:>8.3f}{marker}'
            )

    # ================================================================
    #                      PHASE: VERIFY
    # ================================================================

    def _enter_verify(self):
        """Chuyển sang phase VERIFY."""
        self.phase = self.PHASE_VERIFY
        self.phase_start_time = None
        self.verify_errors = []
        self.verify_pid = PIDController(
            kp=self.computed_kp,
            ki=self.computed_ki,
            kd=self.computed_kd,
            output_min=-self.max_velocity,
            output_max=self.max_velocity,
            integral_max=10.0,
            derivative_filter_alpha=0.1
        )

        self.get_logger().info('')
        self.get_logger().info('━' * 50)
        self.get_logger().info('  Phase 5/5: KIỂM NGHIỆM')
        self.get_logger().info(
            f'  PID: Kp={self.computed_kp:.4f}, '
            f'Ki={self.computed_ki:.4f}, '
            f'Kd={self.computed_kd:.4f}'
        )
        self.get_logger().info(f'  Thời gian kiểm nghiệm: {self.verify_duration}s')
        self.get_logger().info('━' * 50)
        self.publish_status('VERIFY')

    def _do_verify(self, pitch, gyro_y, timestamp):
        """Phase VERIFY: Kiểm nghiệm bộ PID đã tính."""
        if self.phase_start_time is None:
            self.phase_start_time = timestamp

        elapsed = timestamp - self.phase_start_time

        # Áp dụng PID đã tính
        error = pitch - 0.0
        output = self.verify_pid.compute(
            error, timestamp, measured_rate=gyro_y
        )
        self.publish_cmd_vel(output)
        self.publish_data(pitch, error, output, 5.0)

        # Ghi lại sai số
        self.verify_errors.append(abs(error))

        if elapsed > self.verify_duration:
            self._analyze_verification()
            self._print_final_results()
            self._save_results_yaml()
            self.phase = self.PHASE_DONE
            self.publish_status('DONE')

    def _analyze_verification(self):
        """Phân tích kết quả kiểm nghiệm."""
        if not self.verify_errors:
            return

        avg_error = sum(self.verify_errors) / len(self.verify_errors)
        max_error = max(self.verify_errors)

        # Sai số xác lập (20% cuối dữ liệu)
        steady_idx = int(0.8 * len(self.verify_errors))
        steady_errors = self.verify_errors[steady_idx:]
        steady_error = (
            sum(steady_errors) / len(steady_errors)
            if steady_errors else 0
        )

        self.get_logger().info('')
        self.get_logger().info('  === KẾT QUẢ KIỂM NGHIỆM ===')
        self.get_logger().info(
            f'  Sai số trung bình: {math.degrees(avg_error):.3f}°'
        )
        self.get_logger().info(
            f'  Sai số lớn nhất:   {math.degrees(max_error):.3f}°'
        )
        self.get_logger().info(
            f'  Sai số xác lập:    {math.degrees(steady_error):.3f}°'
        )

        # Đánh giá chất lượng
        if steady_error < 0.035:  # < 2 degrees
            self.get_logger().info('  ✓ TUYỆT VỜI! Robot cân bằng rất tốt.')
        elif steady_error < 0.087:  # < 5 degrees
            self.get_logger().info('  ✓ TỐT. Robot cân bằng ổn.')
        elif steady_error < 0.175:  # < 10 degrees
            self.get_logger().info(
                '  ~ TẠM ÔN. Có thể thử rule khác hoặc '
                'điều chỉnh relay_amplitude.'
            )
        else:
            self.get_logger().warn(
                '  ✗ CHƯA TỐT. Thử: '
                '1) Tăng/giảm relay_amplitude, '
                '2) Đổi zn_rule, '
                '3) Tăng num_cycles.'
            )

    # ================================================================
    #                    OUTPUT RESULTS
    # ================================================================

    def _print_final_results(self):
        """In kết quả tuning cuối cùng."""
        self.get_logger().info('')
        self.get_logger().info('=' * 55)
        self.get_logger().info('         KẾT QUẢ TUNING PID HOÀN TẤT')
        self.get_logger().info('=' * 55)
        self.get_logger().info(f'  Ultimate Gain  (Ku) = {self.ku:.6f}')
        self.get_logger().info(f'  Ultimate Period (Tu) = {self.tu:.6f} s')
        self.get_logger().info('')
        self.get_logger().info(f'  ┌─────────────────────────────┐')
        self.get_logger().info(f'  │  Kp = {self.computed_kp:>12.6f}        │')
        self.get_logger().info(f'  │  Ki = {self.computed_ki:>12.6f}        │')
        self.get_logger().info(f'  │  Kd = {self.computed_kd:>12.6f}        │')
        self.get_logger().info(f'  └─────────────────────────────┘')
        self.get_logger().info('')
        self.get_logger().info('  Áp dụng vào controller:')
        self.get_logger().info(
            f'    ros2 param set /balance_controller kp '
            f'{self.computed_kp:.4f}'
        )
        self.get_logger().info(
            f'    ros2 param set /balance_controller ki '
            f'{self.computed_ki:.4f}'
        )
        self.get_logger().info(
            f'    ros2 param set /balance_controller kd '
            f'{self.computed_kd:.4f}'
        )
        self.get_logger().info('=' * 55)
        self.get_logger().info('')

    def _save_results_yaml(self):
        """Lưu kết quả vào file YAML để dùng lại."""
        rule = self.ZN_RULES.get(
            self.zn_rule, self.ZN_RULES['some_overshoot']
        )

        yaml_content = f"""# =============================================
# PID Tuning Results - Kết quả tự động tune PID
# =============================================
# Phương pháp:      Relay Feedback + Ziegler-Nichols
# Quy tắc ZN:       {rule['name']}
# Ultimate Gain:     Ku = {self.ku:.6f}
# Ultimate Period:   Tu = {self.tu:.6f} s
# =============================================

balance_controller:
  ros__parameters:
    kp: {self.computed_kp:.6f}
    ki: {self.computed_ki:.6f}
    kd: {self.computed_kd:.6f}
    target_pitch: 0.0
    max_velocity: 1.5
    integral_max: 10.0
    derivative_filter_alpha: 0.1
    use_gyro_derivative: true
    fall_threshold: 0.785
    enabled: true
"""
        try:
            output_path = os.path.expanduser(self.output_file)
            with open(output_path, 'w') as f:
                f.write(yaml_content)
            self.get_logger().info(
                f'  Kết quả đã lưu: {output_path}'
            )
        except Exception as e:
            self.get_logger().warn(f'  Không thể lưu file: {e}')
            self.get_logger().info('  Nội dung YAML:')
            for line in yaml_content.strip().split('\n'):
                self.get_logger().info(f'    {line}')


def main(args=None):
    rclpy.init(args=args)
    node = PIDTunerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Tuner dừng bởi người dùng.')
    finally:
        # Dừng động cơ
        twist = Twist()
        node.cmd_vel_pub.publish(twist)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
