"""
PID Auto-Tuner Node - Tự động tìm hệ số PID tối ưu cho xe cân bằng hai bánh.

Tính năng nâng cao:
- Tự động Reset thế giới trong Gazebo (/reset_world) qua nhiều vòng lặp (Iterations)
- Phương pháp: Relay Feedback (Åström-Hägglund) tinh chỉnh riêng cho Inverted Pendulum
- Tự động dựng lại robot khi ngã và thử lại
- Đánh giá chất lượng (Score & RMS Error) và chọn ra bộ thông số tối ưu nhất
"""

import math
import os
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, String
from std_srvs.srv import Empty

from balance_robot_controller.pid import PIDController


def quaternion_to_pitch(q):
    """Trích xuất góc pitch từ quaternion."""
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    if abs(sinp) >= 1.0:
        return math.copysign(math.pi / 2.0, sinp)
    return math.asin(sinp)


class PIDTunerNode(Node):
    """
    Node tự động điều chỉnh PID với cơ chế Auto-Reset đa vòng lặp trong Gazebo.
    """

    # === Các phase tuning ===
    PHASE_WAIT = 'WAIT'
    PHASE_RESET = 'RESET'
    PHASE_STABILIZE = 'STABILIZE'
    PHASE_RELAY = 'RELAY'
    PHASE_ANALYZE = 'ANALYZE'
    PHASE_VERIFY = 'VERIFY'
    PHASE_DONE = 'DONE'

    # === Quy tắc điều khiển tinh chỉnh riêng cho Inverted Pendulum ===
    # (Tránh lỗi Ki quá lớn của công thức ZN công nghiệp cổ điển)
    TUNING_RULES = {
        'balance_optimized': {
            'name': 'Balance Robot Optimized (Khuyên dùng)',
            'kp_factor': 0.50,
            'ki_ratio': 0.015,   # Ki = 0.015 * Kp (nhỏ để chống trôi)
            'td_factor': 0.12,   # Kd = Kp * 0.12 * Tu
        },
        'balance_stiff': {
            'name': 'Balance Robot Stiff (Cứng vững)',
            'kp_factor': 0.65,
            'ki_ratio': 0.010,
            'td_factor': 0.15,
        },
        'balance_smooth': {
            'name': 'Balance Robot Smooth (Mềm mại)',
            'kp_factor': 0.40,
            'ki_ratio': 0.020,
            'td_factor': 0.10,
        },
    }

    def __init__(self):
        super().__init__('pid_tuner')

        # ===== Parameters =====
        self.declare_parameter('relay_amplitude', 0.15)      # Biên độ relay nhẹ nhàng (m/s)
        self.declare_parameter('stabilizing_kp', 48.0)       # Kp cơ sở đủ mạnh giữ xe đứng
        self.declare_parameter('stabilizing_kd', 5.0)        # Kd cơ sở giảm chấn
        self.declare_parameter('num_cycles', 3)              # 3 chu kỳ là đủ chính xác và nhanh
        self.declare_parameter('zn_rule', 'balance_optimized')
        self.declare_parameter('stabilize_duration', 1.5)
        self.declare_parameter('verify', True)
        self.declare_parameter('verify_duration', 5.0)
        self.declare_parameter('fall_threshold', 0.785)
        self.declare_parameter('max_velocity', 1.5)
        self.declare_parameter('max_iterations', 3)          # 3 lần lấy mẫu tối ưu
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
        self.max_iterations = self.get_parameter('max_iterations').value
        self.output_file = self.get_parameter('output_file').value

        # Trạng thái vòng lặp
        self.current_iteration = 1
        self.iteration_results = []

        # Trạng thái tuner
        self.phase = self.PHASE_WAIT
        self.phase_start_time = None
        self.start_time = None

        # Dữ liệu relay
        self.relay_data = []
        self.zero_crossings = []
        self.peaks = []
        self.prev_sign = 0
        self.cycle_count = 0
        self.half_cycle_max_pitch = 0.0

        # Kết quả tuning
        self.ku = 0.0
        self.tu = 0.0
        self.computed_kp = 0.0
        self.computed_ki = 0.0
        self.computed_kd = 0.0

        # Verification
        self.verify_pid = None
        self.verify_errors = []

        # Service clients reset Gazebo
        self.reset_world_cli = self.create_client(Empty, '/reset_world')
        self.reset_sim_cli = self.create_client(Empty, '/reset_simulation')

        # ===== Publishers =====
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, 'pid_tuner/status', 10)
        self.data_pub = self.create_publisher(
            Float64MultiArray, 'pid_tuner/data', 10
        )

        # ===== Subscriber (SensorDataQoS chống rớt gói) =====
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, qos_profile_sensor_data
        )

        # ===== Log khởi động =====
        self.get_logger().info('')
        self.get_logger().info('=' * 60)
        self.get_logger().info('  🤖 PID AUTO-TUNER VỚI TỰ ĐỘNG RESET GAZEBO')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'  Số lần lấy mẫu tối ưu: {self.max_iterations} lần')
        self.get_logger().info(f'  Biên độ kích thích:     {self.relay_amplitude} m/s')
        self.get_logger().info(f'  Quy tắc tối ưu:        {self.zn_rule}')
        self.get_logger().info('=' * 60)
        self.get_logger().info('Đang chờ dữ liệu IMU để bắt đầu...')
        self.get_logger().info('')

    def trigger_gazebo_reset(self):
        """Gọi service reset trong Gazebo để robot đứng thẳng lại."""
        req = Empty.Request()
        if self.reset_world_cli.service_is_ready():
            self.reset_world_cli.call_async(req)
            self.get_logger().info('🔄 Đã gọi Gazebo /reset_world để dựng lại robot!')
        elif self.reset_sim_cli.service_is_ready():
            self.reset_sim_cli.call_async(req)
            self.get_logger().info('🔄 Đã gọi Gazebo /reset_simulation để dựng lại robot!')
        else:
            self.get_logger().warn('⚠️ Gazebo reset service đang chờ kết nối...')

    def publish_status(self, text):
        """Publish trạng thái tuner."""
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def publish_data(self, pitch, error, output, phase_id):
        """Publish dữ liệu cho monitoring."""
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

    def imu_callback(self, msg):
        """State machine chính xử lý theo phase và vòng lặp."""
        pitch = quaternion_to_pitch(msg.orientation)
        gyro_y = msg.angular_velocity.y
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.start_time is None:
            self.start_time = timestamp

        # Phát hiện ngã -> Tự động gọi Gazebo reset và thử lại
        if abs(pitch) > self.fall_threshold:
            if self.phase not in [self.PHASE_DONE, self.PHASE_WAIT, self.PHASE_RESET]:
                self.get_logger().warn(
                    f'⚠️ Robot ngã ở lần thử #{self.current_iteration} (Pitch = {math.degrees(pitch):.1f}°)! '
                    f'Tự động reset Gazebo và thử lại...'
                )
                self.publish_cmd_vel(0.0)
                self.stab_kp = min(70.0, self.stab_kp * 1.1)
                self.stab_kd = min(8.0, self.stab_kd * 1.1)
                self._enter_reset(timestamp)
                return

        # State machine
        if self.phase == self.PHASE_WAIT:
            self._enter_reset(timestamp)

        elif self.phase == self.PHASE_RESET:
            self._do_reset(pitch, gyro_y, timestamp)

        elif self.phase == self.PHASE_STABILIZE:
            self._do_stabilize(pitch, gyro_y, timestamp)

        elif self.phase == self.PHASE_RELAY:
            self._do_relay(pitch, gyro_y, timestamp)

        elif self.phase == self.PHASE_ANALYZE:
            self._do_analyze(timestamp)

        elif self.phase == self.PHASE_VERIFY:
            self._do_verify(pitch, gyro_y, timestamp)

        elif self.phase == self.PHASE_DONE:
            self.publish_cmd_vel(0.0)

    # ================================================================
    #                     PHASE: RESET
    # ================================================================

    def _enter_reset(self, timestamp):
        """Chuyển sang phase RESET và gọi Gazebo reset."""
        self.phase = self.PHASE_RESET
        self.phase_start_time = timestamp
        self.trigger_gazebo_reset()
        self.publish_status(f'RESETTING (Lần {self.current_iteration}/{self.max_iterations})')

    def _do_reset(self, pitch, gyro_y, timestamp):
        """Giữ robot thăng bằng ngay lập tức trong khi Gazebo ổn định thế giới."""
        error = pitch - 0.0
        output = self.stab_kp * error + self.stab_kd * gyro_y
        self.publish_cmd_vel(output)
        if timestamp - self.phase_start_time > 0.8:
            self._enter_stabilize(timestamp)

    # ================================================================
    #                     PHASE: STABILIZE
    # ================================================================

    def _enter_stabilize(self, timestamp):
        """Chuyển sang phase STABILIZE."""
        self.phase = self.PHASE_STABILIZE
        self.phase_start_time = timestamp
        self.get_logger().info('')
        self.get_logger().info('━' * 55)
        self.get_logger().info(f'  📍 LẦN LẤY MẪU #{self.current_iteration}/{self.max_iterations}: ỔN ĐỊNH')
        self.get_logger().info(f'  Sử dụng PD cơ sở: Kp={self.stab_kp:.1f}, Kd={self.stab_kd:.1f}')
        self.get_logger().info('━' * 55)
        self.publish_status(f'STABILIZE (#{self.current_iteration})')

    def _do_stabilize(self, pitch, gyro_y, timestamp):
        """Phase STABILIZE: Ổn định robot trước khi tạo dao động."""
        elapsed = timestamp - self.phase_start_time

        error = pitch - 0.0
        output = self.stab_kp * error + self.stab_kd * gyro_y
        self.publish_cmd_vel(output)
        self.publish_data(pitch, error, output, 1.0)

        if elapsed > self.stabilize_duration:
            if abs(pitch) < 0.1:  # < 6 độ
                self.get_logger().info(
                    f'  ✓ Robot đã đứng thẳng ổn định! (pitch = {math.degrees(pitch):.2f}°)'
                )
                self._enter_relay(timestamp)
            else:
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

        # Điều chỉnh biên độ relay theo từng vòng lặp để lấy mẫu đa dạng
        if self.current_iteration == 1:
            self.active_amplitude = self.relay_amplitude * 0.85
        elif self.current_iteration == 2:
            self.active_amplitude = self.relay_amplitude
        else:
            self.active_amplitude = self.relay_amplitude * 1.15

        self.get_logger().info('  📍 KÍCH HOẠT DAO ĐỘNG RELAY...')
        self.get_logger().info(f'  Biên độ kích thích: {self.active_amplitude:.3f} m/s')
        self.publish_status(f'RELAY (#{self.current_iteration})')

    def _do_relay(self, pitch, gyro_y, timestamp):
        """Phase RELAY: Tạo dao động có kiểm soát."""
        error = pitch - 0.0
        stab_output = self.stab_kp * error + self.stab_kd * gyro_y

        if error > 0:
            relay_output = self.active_amplitude
        else:
            relay_output = -self.active_amplitude

        output = stab_output + relay_output
        self.publish_cmd_vel(output)
        self.publish_data(pitch, error, output, 2.0)

        self.relay_data.append((timestamp, pitch, output))
        self.half_cycle_max_pitch = max(self.half_cycle_max_pitch, abs(pitch))

        current_sign = 1 if pitch >= 0 else -1
        if self.prev_sign == 0:
            self.prev_sign = current_sign
            return

        if current_sign != self.prev_sign:
            self.zero_crossings.append(timestamp)
            self.cycle_count += 1
            if self.half_cycle_max_pitch > 0.0001:
                self.peaks.append(self.half_cycle_max_pitch)
            self.half_cycle_max_pitch = 0.0

        self.prev_sign = current_sign

        if self.cycle_count >= self.num_cycles * 2:
            self.get_logger().info(f'  ✓ Đã đo đủ {self.num_cycles} chu kỳ dao động!')
            self.phase = self.PHASE_ANALYZE

    # ================================================================
    #                     PHASE: ANALYZE
    # ================================================================

    def _do_analyze(self, timestamp):
        """Phase ANALYZE: Tính Ku, Tu và bộ PID."""
        self.publish_status(f'ANALYZE (#{self.current_iteration})')

        if len(self.zero_crossings) < 4:
            self.get_logger().error('  ✗ Không đủ dữ liệu zero-crossings, reset thử lại...')
            self._enter_reset(timestamp)
            return

        crossings = self.zero_crossings[2:]
        half_periods = [crossings[i] - crossings[i - 1] for i in range(1, len(crossings)) if crossings[i] - crossings[i - 1] > 0.001]

        if not half_periods:
            self._enter_reset(timestamp)
            return

        avg_half_period = sum(half_periods) / len(half_periods)
        self.tu = 2.0 * avg_half_period

        valid_peaks = self.peaks[1:] if len(self.peaks) > 1 else self.peaks
        avg_amplitude = sum(valid_peaks) / len(valid_peaks)

        # Tính Ku = 4*d / (pi*a)
        self.ku = 4.0 * self.active_amplitude / (math.pi * avg_amplitude)

        # Tính thông số PID theo quy tắc xe cân bằng
        rule = self.TUNING_RULES.get(self.zn_rule, self.TUNING_RULES['balance_optimized'])
        self.computed_kp = rule['kp_factor'] * self.ku
        self.computed_kd = self.computed_kp * rule['td_factor'] * self.tu
        self.computed_ki = rule['ki_ratio'] * self.computed_kp

        self.get_logger().info(f'  Ku = {self.ku:.4f}, Tu = {self.tu:.4f}s')
        self.get_logger().info(
            f'  >> Tính được: Kp={self.computed_kp:.3f} | Ki={self.computed_ki:.3f} | Kd={self.computed_kd:.3f}'
        )

        if self.do_verify:
            self._enter_verify()
        else:
            self._finish_all_iterations()

    # ================================================================
    #                      PHASE: VERIFY
    # ================================================================

    def _enter_verify(self):
        """Chuyển sang phase VERIFY kiểm nghiệm PID vừa tính."""
        self.phase = self.PHASE_VERIFY
        self.phase_start_time = None
        self.verify_errors = []
        self.verify_pitches = []
        self.verify_outputs = []
        self.verify_pid = PIDController(
            kp=self.computed_kp,
            ki=self.computed_ki,
            kd=self.computed_kd,
            output_min=-self.max_velocity,
            output_max=self.max_velocity,
            integral_max=5.0,
            derivative_filter_alpha=0.1
        )
        self.get_logger().info(f'  📍 KIỂM NGHIỆM ĐỘ ỔN ĐỊNH ({self.verify_duration}s)...')
        self.publish_status(f'VERIFY (#{self.current_iteration})')

    def _do_verify(self, pitch, gyro_y, timestamp):
        """Phase VERIFY: Chạy thử và chấm điểm."""
        if self.phase_start_time is None:
            self.phase_start_time = timestamp

        elapsed = timestamp - self.phase_start_time

        error = pitch - 0.0
        output = self.verify_pid.compute(error, timestamp, measured_rate=gyro_y)
        self.publish_cmd_vel(output)
        self.publish_data(pitch, error, output, 5.0)

        self.verify_errors.append(abs(error))
        self.verify_pitches.append(pitch)
        self.verify_outputs.append(output)

        if elapsed > self.verify_duration:
            # Chấm điểm chất lượng
            avg_err = sum(self.verify_errors) / len(self.verify_errors)
            rms_err = math.sqrt(sum(e**2 for e in self.verify_errors) / len(self.verify_errors))
            max_err = max(self.verify_errors)
            score = 100.0 / (1.0 + 25.0 * rms_err)

            # Tự động tính góc cân bằng tự nhiên (triệt tiêu trôi)
            steady_idx = int(0.5 * len(self.verify_pitches))
            steady_pitches = self.verify_pitches[steady_idx:]
            steady_outputs = self.verify_outputs[steady_idx:]
            avg_pitch = sum(steady_pitches) / len(steady_pitches) if steady_pitches else 0.0
            avg_output = sum(steady_outputs) / len(steady_outputs) if steady_outputs else 0.0
            calibrated_target_pitch = avg_pitch + (0.005 * avg_output)

            res = {
                'iteration': self.current_iteration,
                'kp': self.computed_kp,
                'ki': self.computed_ki,
                'kd': self.computed_kd,
                'target_pitch': calibrated_target_pitch,
                'avg_deg': math.degrees(avg_err),
                'rms_deg': math.degrees(rms_err),
                'max_deg': math.degrees(max_err),
                'score': score
            }
            self.iteration_results.append(res)
            self.get_logger().info(
                f'  ✓ Lần #{self.current_iteration}: Sai số RMS = {res["rms_deg"]:.2f}°, '
                f'Target Pitch tối ưu = {calibrated_target_pitch:.4f} rad, Điểm = {score:.1f}/100'
            )

            if self.current_iteration < self.max_iterations:
                self.current_iteration += 1
                self.get_logger().info(f'🔄 Tự động reset Gazebo để bắt đầu lần #{self.current_iteration}...')
                self._enter_reset(timestamp)
            else:
                self._finish_all_iterations()

    # ================================================================
    #                    HOÀN TẤT & LỰA CHỌN BỘ TỐI ƯU
    # ================================================================

    def _finish_all_iterations(self):
        """Tổng kết tất cả các lần thử, chọn ra bộ thông số điểm cao nhất."""
        self.phase = self.PHASE_DONE
        self.publish_cmd_vel(0.0)
        self.publish_status('DONE')

        if not self.iteration_results:
            self.get_logger().warn('Chưa có kết quả vòng lặp nào được ghi nhận.')
            return

        # Chọn kết quả có score cao nhất (sai số thấp nhất)
        best_res = max(self.iteration_results, key=lambda x: x['score'])

        self.get_logger().info('')
        self.get_logger().info('=' * 75)
        self.get_logger().info('           🏆 BẢNG TỔNG KẾT TỰ ĐỘNG TỐI ƯU HÓA PID 🏆')
        self.get_logger().info('=' * 75)
        self.get_logger().info(f' {"Lần":<4} | {"Kp":<7} | {"Ki":<6} | {"Kd":<6} | {"Target Pitch":<14} | {"RMS Error":<10} | {"Điểm":<5}')
        self.get_logger().info('-' * 75)
        for r in self.iteration_results:
            is_best = ' ⭐ (TỐT NHẤT)' if r == best_res else ''
            self.get_logger().info(
                f' #{r["iteration"]:<3} | {r["kp"]:<7.2f} | {r["ki"]:<6.3f} | {r["kd"]:<6.3f} | '
                f'{r["target_pitch"]:>+.5f} rad  | {r["rms_deg"]:<8.2f}° | {r["score"]:<5.1f}{is_best}'
            )
        self.get_logger().info('=' * 75)
        self.get_logger().info('')
        self.get_logger().info(f'  🎯 BỘ THÔNG SỐ TỐI ƯU NHẤT (ĐỨNG VỮNG, KHÔNG TRÔI):')
        self.get_logger().info(f'    Kp           = {best_res["kp"]:.4f}')
        self.get_logger().info(f'    Ki           = {best_res["ki"]:.4f}')
        self.get_logger().info(f'    Kd           = {best_res["kd"]:.4f}')
        self.get_logger().info(f'    Target Pitch = {best_res["target_pitch"]:.5f} rad ({math.degrees(best_res["target_pitch"]):.3f}°)')
        self.get_logger().info('')

        # Lưu vào file YAML
        self._save_results_yaml(best_res)

    def _save_results_yaml(self, best):
        """Lưu bộ số tốt nhất ra file YAML."""
        yaml_content = f"""# =============================================
# PID Tuning Results - Kết quả tự động tối ưu hóa qua {self.max_iterations} lần lấy mẫu
# Điểm đánh giá: {best['score']:.1f}/100 | Sai số RMS: {best['rms_deg']:.2f} độ
# Góc cân bằng tự nhiên (chống trôi): {best['target_pitch']:.5f} rad
# =============================================

balance_controller:
  ros__parameters:
    kp: {best['kp']:.4f}
    ki: {best['ki']:.4f}
    kd: {best['kd']:.4f}
    target_pitch: {best['target_pitch']:.5f}
    max_velocity: 1.5
    integral_max: 5.0
    derivative_filter_alpha: 0.1
    use_gyro_derivative: true
    fall_threshold: 0.785
    enabled: true
"""
        try:
            output_path = os.path.expanduser(self.output_file)
            with open(output_path, 'w') as f:
                f.write(yaml_content)
            self.get_logger().info(f'  📁 Đã lưu file cấu hình tối ưu tại: {output_path}')
        except Exception as e:
            self.get_logger().warn(f'  Không thể lưu file: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = PIDTunerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Tuner dừng bởi người dùng.')
    finally:
        twist = Twist()
        node.cmd_vel_pub.publish(twist)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
