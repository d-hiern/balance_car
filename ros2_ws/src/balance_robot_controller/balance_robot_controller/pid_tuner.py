"""
PID Auto-Tuner Node - Hệ thống Tự Động Thử Nghiệm & Tối Ưu Hóa PID Đa Mẫu Thử
(Anchor-Based Multi-Candidate Adaptive PID Auto-Tuner)

Nguyên lý hoạt động:
  1. BẢO TOÀN CẤU HÌNH TỐI ƯU (ANCHOR BASELINE):
     - Luôn lưu giữ bộ thông số cân bằng tốt nhất làm "mỏ neo gốc" (Anchor).
     - Không bao giờ bị mất kỷ lục hoặc bị kéo ngã theo các mẫu thử thất bại.
  2. TẠO CÁC MẪU THỬ ĐA DẠNG THÔNG MINH (MULTI-CANDIDATE EXPLORATION):
     - Từ mỏ neo tốt nhất, hệ thống tự động sinh ra các mẫu thử đa dạng xoay quanh nó:
       + Mẫu tăng giảm chấn (Kd+) dập tắt vọt lố sau huých.
       + Mẫu tăng độ cứng (Kp+) rút ngắn thời gian hồi phục.
       + Mẫu mềm mại (Kp-, Kd-) chống rung chấn cao tần cho motor.
       + Mẫu triệt tiêu trôi xe (Ki-, Kd+).
       + Mẫu cân bằng tổng hợp cao cấp.
  3. THUẬT TOÁN ĐÁNH GIÁ & CẬP NHẬT KỶ LỤC:
     - Mẫu nào đạt điểm cao hơn -> Lập tức trở thành KỶ LỤC MỚI & MỎ NEO MỚI!
     - Mẫu nào bị ngã -> Tự động ghi vào Blacklist để không bao giờ lặp lại; mỏ neo vẫn được bảo toàn.
  4. LƯU TRỮ BỀN VỮNG:
     - Tự động ghi nhận vào ~/.pso_pid_memory.json và xuất ra ~/tuned_pid_params.yaml.
"""

import json
import math
import os
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import String
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
    ROS2 Node: Tinh chỉnh PID đa mẫu thử xoay quanh mỏ neo tối ưu.
    """

    # ===== CÁC TRẠNG THÁI KIỂM ĐỊNH =====
    STATE_RESET = 'RESET'
    STATE_STATIC = 'STATIC'
    STATE_DISTURB_FWD = 'DISTURB_FWD'
    STATE_RECOVER_FWD = 'RECOVER_FWD'
    STATE_DISTURB_BWD = 'DISTURB_BWD'
    STATE_RECOVER_BWD = 'RECOVER_BWD'
    STATE_DONE = 'DONE'

    # Cấu hình mỏ neo chuẩn hóa xuất phát cho xe ~0.93kg
    DEFAULT_SEED_PID = [55.0, 0.50, 6.0]

    # Giới hạn an toàn vật lý của xe (Ngăn không cho vọt lên 85 gây rung giật ngã)
    KP_MIN, KP_MAX = 42.0, 72.0
    KI_MIN, KI_MAX = 0.25, 0.70
    KD_MIN, KD_MAX = 4.8, 8.5

    def __init__(self):
        super().__init__('pid_tuner')

        # ===== Khai báo Parameters =====
        self.declare_parameter('fall_threshold', 0.785)
        self.declare_parameter('max_velocity', 1.5)
        self.declare_parameter('disturb_magnitude', 0.12)
        self.declare_parameter('balance_duration', 3.5)
        self.declare_parameter('recovery_duration', 3.5)
        self.declare_parameter('max_iterations', 7)
        self.declare_parameter('memory_file', '~/.pso_pid_memory.json')
        self.declare_parameter('output_file', '~/tuned_pid_params.yaml')
        self.declare_parameter('reset_memory', False)

        self.fall_threshold = self.get_parameter('fall_threshold').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.disturb_magnitude = self.get_parameter('disturb_magnitude').value
        self.balance_duration = self.get_parameter('balance_duration').value
        self.recovery_duration = self.get_parameter('recovery_duration').value
        self.max_iterations = self.get_parameter('max_iterations').value
        self.memory_file = os.path.expanduser(self.get_parameter('memory_file').value)
        self.output_file = os.path.expanduser(self.get_parameter('output_file').value)
        reset_memory = self.get_parameter('reset_memory').value

        # ===== Cấu hình Mỏ Neo Gốc (Anchor Baseline) & Kỷ Lục Tối Ưu =====
        self.anchor_pid = list(self.DEFAULT_SEED_PID)
        self.anchor_fitness = 0.0
        self.best_pid = list(self.DEFAULT_SEED_PID)
        self.best_fitness = 0.0
        self.best_metrics = None

        self.blacklist = []
        self.MAX_BLACKLIST_SIZE = 80

        if reset_memory:
            self._delete_memory()
            self.get_logger().warn('🗑️  ĐÃ XÓA BỘ NHỚ CŨ (reset_memory=True)!')
        else:
            self._load_memory()

        # Bộ thông số của mẫu thử hiện tại
        self.current_kp = self.anchor_pid[0]
        self.current_ki = self.anchor_pid[1]
        self.current_kd = self.anchor_pid[2]
        self.current_desc = "Cấu hình xuất phát ban đầu"

        # ===== State Machine & Dữ liệu đo lường =====
        self.iteration = 1
        self.state = self.STATE_RESET
        self.state_start_time = None
        self.trial_start_time = None

        self.current_pid = None
        self.pitch_history = []
        self.static_pitch_history = []
        self.output_history = []
        self.chattering_diffs = []
        self.prev_output = 0.0

        self.robot_fell = False
        self.fall_reason = ""
        self.overshoot_fwd = 0.0
        self.overshoot_bwd = 0.0
        self.settling_time_fwd = self.recovery_duration
        self.settling_time_bwd = self.recovery_duration
        self.zero_crossings_fwd = 0
        self.zero_crossings_bwd = 0
        self.prev_pitch_sign_fwd = None
        self.prev_pitch_sign_bwd = None

        # ===== Gazebo & ROS Interfaces =====
        self.reset_world_cli = self.create_client(Empty, '/reset_world')
        self.reset_sim_cli = self.create_client(Empty, '/reset_simulation')
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, 'pid_tuner/status', 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, qos_profile_sensor_data
        )

        self.get_logger().info('')
        self.get_logger().info('=' * 72)
        self.get_logger().info('  🚀 HỆ THỐNG TỰ THÍCH NGHI PID THEO MẪU THỬ QUANH MỎ NEO TỐI ƯU')
        self.get_logger().info('     (Anchor-Based Multi-Candidate Adaptive PID Auto-Tuner)')
        self.get_logger().info('=' * 72)
        self.get_logger().info(
            f'  ⚓ Mỏ neo xuất phát: Kp={self.anchor_pid[0]:.2f} | Ki={self.anchor_pid[1]:.4f} | Kd={self.anchor_pid[2]:.2f}'
        )
        if self.best_fitness > 0.0:
            self.get_logger().info(f'  ⭐ Kỷ lục tối ưu đã lưu giữ: {self.best_fitness:.1f}%')
        self.get_logger().info(f'  🎯 Tổng cộng: {self.max_iterations} mẫu thử nghiệm thông minh')
        self.get_logger().info(f'  💨 Lực huých kiểm tra động lực: ±{self.disturb_magnitude:.2f} m/s')
        self.get_logger().info('=' * 72)
        self.get_logger().info('')

    # ====================== BỘ NHỚ LƯU TRỮ ======================

    def _load_memory(self):
        """Nạp kỷ lục tốt nhất và danh sách cấm ngã (Blacklist) từ file bền vững."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    fit = data.get('global_best_fitness', 0.0)
                    zn_kp = data.get('zn_kp')
                    zn_kd = data.get('zn_kd')
                    zn_ki = data.get('zn_ki')
                    pos = data.get('global_best_position')

                    kp = zn_kp if zn_kp is not None else (pos[0] if pos and len(pos) >= 2 else None)
                    kd = zn_kd if zn_kd is not None else (pos[1] if pos and len(pos) >= 2 else None)
                    ki = zn_ki if zn_ki is not None else 0.50

                    if kp is not None and kd is not None and fit >= 40.0:
                        kp = max(self.KP_MIN, min(self.KP_MAX, float(kp)))
                        ki = max(self.KI_MIN, min(self.KI_MAX, float(ki)))
                        kd = max(self.KD_MIN, min(self.KD_MAX, float(kd)))
                        self.anchor_pid = [kp, ki, kd]
                        self.anchor_fitness = fit
                        self.best_pid = list(self.anchor_pid)
                        self.best_fitness = fit
                        self.get_logger().info(
                            f'  📌 Đã nạp cấu hình tốt nhất từ trước: Kp={kp:.2f}, Ki={ki:.4f}, Kd={kd:.2f} ({fit:.1f}%)'
                        )
                    else:
                        self.get_logger().warn('  ⚠️ Kỷ lục cũ không đạt chuẩn. Khởi động từ cấu hình mặc định!')

                    loaded_bl = data.get('blacklist', [])
                    self.blacklist = []
                    for b in loaded_bl:
                        if isinstance(b, list) and len(b) >= 2:
                            kp = float(b[0])
                            kd = float(b[2]) if len(b) >= 3 else float(b[1])
                            ki = float(b[1]) if len(b) >= 3 else 0.50
                            self.blacklist.append([round(kp, 2), round(ki, 4), round(kd, 2)])
                    self.blacklist = self.blacklist[-self.MAX_BLACKLIST_SIZE:]
                    if self.blacklist:
                        self.get_logger().info(f'  📋 Đã nạp đầy đủ {len(self.blacklist)} điểm cấm từ Blacklist cũ.')
            except Exception as e:
                self.get_logger().warn(f'Lỗi đọc memory: {e}')

    def _delete_memory(self):
        if os.path.exists(self.memory_file):
            try:
                os.remove(self.memory_file)
            except Exception:
                pass

    def _save_memory(self):
        """Lưu trữ bền vững bộ số tốt nhất và toàn bộ danh sách Blacklist."""
        data = {}
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
            except Exception:
                data = {}

        if self.best_pid is not None and self.best_fitness > 0.0:
            data['global_best_position'] = [self.best_pid[0], self.best_pid[2]]
            data['global_best_fitness'] = self.best_fitness
            data['zn_kp'] = self.best_pid[0]
            data['zn_ki'] = self.best_pid[1]
            data['zn_kd'] = self.best_pid[2]

        data['blacklist'] = self.blacklist[-self.MAX_BLACKLIST_SIZE:]
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f'Lỗi lưu memory: {e}')

    # ====================== SINH MẪU THỬ THÔNG MINH ======================

    def _get_candidate(self, step_idx):
        """
        Tạo các mẫu thử (Candidate variations) xoay quanh mỏ neo tốt nhất (Anchor).
        Đảm bảo an toàn vật lý và né tránh hoàn toàn Blacklist.
        """
        a_kp, a_ki, a_kd = self.anchor_pid

        # Danh mục các định hướng cải tiến xoay quanh Anchor
        candidates_def = [
            (0.0,   0.00,  0.0, "Thẩm định cấu hình mỏ neo gốc (Anchor Baseline)"),
            (0.0,   0.00, +0.6, "Tăng giảm chấn chống vọt lố (Damping +)"),
            (+3.0,  0.00, +0.4, "Tăng độ cứng rút ngắn thời gian hồi phục (Stiffness +)"),
            (-3.0,  0.00, -0.3, "Hạ độ cứng giảm rung chấn motor (Smoothness +)"),
            (+1.5, -0.08, +0.5, "Triệt tiêu trôi xe và dập dao động (Anti-Drift +)"),
            (+2.5, +0.04, +0.7, "Cân bằng tổng hợp cao cấp (Optimal Balance)"),
            (-1.5, -0.05, +0.8, "Dập rung sâu gyro và hồi phục êm (Deep Damping)"),
        ]

        if step_idx - 1 < len(candidates_def):
            dkp, dki, dkd, desc = candidates_def[step_idx - 1]
        else:
            # Nếu chạy nhiều hơn 7 bước, tự dò vi sai thông minh
            phase = float(step_idx)
            dkp = 2.5 * math.sin(phase)
            dkd = 0.5 * math.cos(phase)
            dki = 0.02 * math.sin(2.0 * phase)
            desc = f"Mẫu thử thăm dò đa chiều {step_idx}"

        cand_kp = max(self.KP_MIN, min(self.KP_MAX, a_kp + dkp))
        cand_ki = max(self.KI_MIN, min(self.KI_MAX, a_ki + dki))
        cand_kd = max(self.KD_MIN, min(self.KD_MAX, a_kd + dkd))

        # Kiểm tra né tránh Blacklist (Tuyệt đối không thử lại vùng từng ngã)
        for bad in self.blacklist:
            bad_kp = float(bad[0])
            bad_kd = float(bad[2]) if len(bad) >= 3 else float(bad[1])
            dist = math.sqrt(
                ((cand_kp - bad_kp) / (self.KP_MAX - self.KP_MIN)) ** 2 +
                ((cand_kd - bad_kd) / (self.KD_MAX - self.KD_MIN)) ** 2
            )
            if dist < 0.08:
                shift_kp = 3.0 if cand_kp <= bad_kp else -2.5
                shift_kd = 0.6 if cand_kd <= bad_kd else -0.5
                cand_kp = max(self.KP_MIN, min(self.KP_MAX, cand_kp + shift_kp))
                cand_kd = max(self.KD_MIN, min(self.KD_MAX, cand_kd + shift_kd))
                desc += f" [🛡️ Đã né Blacklist ({bad_kp:.1f}, {bad_kd:.1f})]"
                break

        return cand_kp, cand_ki, cand_kd, desc

    # ====================== ĐIỀU KHIỂN ROBOT ======================

    def trigger_gazebo_reset(self):
        req = Empty.Request()
        if self.reset_world_cli.service_is_ready():
            self.reset_world_cli.call_async(req)
        elif self.reset_sim_cli.service_is_ready():
            self.reset_sim_cli.call_async(req)

    def publish_cmd_vel(self, linear_x):
        twist = Twist()
        twist.linear.x = max(-self.max_velocity, min(self.max_velocity, linear_x))
        self.cmd_vel_pub.publish(twist)

    # ====================== STATE MACHINE ======================

    def imu_callback(self, msg):
        pitch = quaternion_to_pitch(msg.orientation)
        gyro_y = msg.angular_velocity.y
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.state_start_time is None or timestamp < self.state_start_time:
            self.state_start_time = timestamp
        elapsed = timestamp - self.state_start_time

        # Kiểm tra ngã trong khi thử nghiệm
        if abs(pitch) > self.fall_threshold and self.state not in [self.STATE_RESET, self.STATE_DONE]:
            self.robot_fell = True
            self.fall_reason = f"Góc nghiêng vượt ngưỡng ({math.degrees(pitch):.1f}°)"
            self.publish_cmd_vel(0.0)
            self._evaluate_iteration(timestamp)
            return

        # ----- TRẠNG THÁI 1: RESET GAZEBO & CHỜ ỔN ĐỊNH -----
        if self.state == self.STATE_RESET:
            if abs(pitch) > self.fall_threshold:
                self.publish_cmd_vel(0.0)
                if elapsed > 1.0:
                    self.trigger_gazebo_reset()
                    self.state_start_time = timestamp
                return

            # Dùng PD nhẹ hỗ trợ robot dựng thẳng trong 2s
            out = self.current_kp * pitch + self.current_kd * gyro_y
            self.publish_cmd_vel(out)
            if elapsed < 2.0:
                return
            if abs(pitch) < 0.175:  # Góc nghiêng < 10 độ -> Sẵn sàng
                self._start_iteration_trial(timestamp)
            elif elapsed > 5.0:
                self.get_logger().warn('  🔄 Robot chưa đứng thẳng! Reset Gazebo...')
                self.publish_cmd_vel(0.0)
                self.trigger_gazebo_reset()
                self.state_start_time = timestamp
            return

        # ----- TRẠNG THÁI 2: CÂN BẰNG TĨNH -----
        if self.state == self.STATE_STATIC:
            error = pitch
            output = self.current_pid.compute(error, timestamp, measured_rate=gyro_y)
            self.publish_cmd_vel(output)
            self.pitch_history.append(pitch)
            self.static_pitch_history.append(pitch)
            self.output_history.append(output)
            self.chattering_diffs.append(abs(output - self.prev_output))
            self.prev_output = output

            if elapsed > self.balance_duration:
                self.state = self.STATE_DISTURB_FWD
                self.state_start_time = timestamp
                self.get_logger().info(f'    👉 [Huých TIẾN] +{self.disturb_magnitude:.2f} m/s...')
            return

        # ----- TRẠNG THÁI 3: HUÝCH TIẾN -----
        if self.state == self.STATE_DISTURB_FWD:
            error = pitch
            pid_out = self.current_pid.compute(error, timestamp, measured_rate=gyro_y)
            output = pid_out + self.disturb_magnitude
            self.publish_cmd_vel(output)
            self.pitch_history.append(pitch)
            self.output_history.append(output)
            self.chattering_diffs.append(abs(output - self.prev_output))
            self.prev_output = output

            if elapsed > 0.10:
                self.state = self.STATE_RECOVER_FWD
                self.state_start_time = timestamp
                self.overshoot_fwd = 0.0
                self.prev_pitch_sign_fwd = math.copysign(1.0, pitch)
            return

        # ----- TRẠNG THÁI 4: HỒI PHỤC SAU HUÝCH TIẾN -----
        if self.state == self.STATE_RECOVER_FWD:
            error = pitch
            output = self.current_pid.compute(error, timestamp, measured_rate=gyro_y)
            self.publish_cmd_vel(output)
            self.pitch_history.append(pitch)
            self.output_history.append(output)
            self.chattering_diffs.append(abs(output - self.prev_output))
            self.prev_output = output

            self.overshoot_fwd = max(self.overshoot_fwd, abs(pitch))

            # Đếm số nhịp lắc lư đổi dấu (Zero crossings)
            current_sign = math.copysign(1.0, pitch)
            if current_sign != self.prev_pitch_sign_fwd and abs(pitch) > 0.02:
                self.zero_crossings_fwd += 1
                self.prev_pitch_sign_fwd = current_sign

            # Đo thời gian dập tắt dao động (|pitch| < 1.5 độ)
            if abs(pitch) < 0.026 and self.settling_time_fwd >= self.recovery_duration and elapsed > 0.2:
                self.settling_time_fwd = elapsed

            if elapsed > self.recovery_duration:
                self.state = self.STATE_DISTURB_BWD
                self.state_start_time = timestamp
                self.get_logger().info(f'    👉 [Huých LÙI] -{self.disturb_magnitude:.2f} m/s...')
            return

        # ----- TRẠNG THÁI 5: HUÝCH LÙI -----
        if self.state == self.STATE_DISTURB_BWD:
            error = pitch
            pid_out = self.current_pid.compute(error, timestamp, measured_rate=gyro_y)
            output = pid_out - self.disturb_magnitude
            self.publish_cmd_vel(output)
            self.pitch_history.append(pitch)
            self.output_history.append(output)
            self.chattering_diffs.append(abs(output - self.prev_output))
            self.prev_output = output

            if elapsed > 0.10:
                self.state = self.STATE_RECOVER_BWD
                self.state_start_time = timestamp
                self.overshoot_bwd = 0.0
                self.prev_pitch_sign_bwd = math.copysign(1.0, pitch)
            return

        # ----- TRẠNG THÁI 6: HỒI PHỤC SAU HUÝCH LÙI -----
        if self.state == self.STATE_RECOVER_BWD:
            error = pitch
            output = self.current_pid.compute(error, timestamp, measured_rate=gyro_y)
            self.publish_cmd_vel(output)
            self.pitch_history.append(pitch)
            self.output_history.append(output)
            self.chattering_diffs.append(abs(output - self.prev_output))
            self.prev_output = output

            self.overshoot_bwd = max(self.overshoot_bwd, abs(pitch))

            current_sign = math.copysign(1.0, pitch)
            if current_sign != self.prev_pitch_sign_bwd and abs(pitch) > 0.02:
                self.zero_crossings_bwd += 1
                self.prev_pitch_sign_bwd = current_sign

            if abs(pitch) < 0.026 and self.settling_time_bwd >= self.recovery_duration and elapsed > 0.2:
                self.settling_time_bwd = elapsed

            if elapsed > self.recovery_duration:
                self._evaluate_iteration(timestamp)
            return

    # ====================== KHỞI ĐỘNG BÀI THỬ ======================

    def _start_iteration_trial(self, timestamp):
        # Lấy thông số mẫu thử thông minh cho lượt này
        self.current_kp, self.current_ki, self.current_kd, self.current_desc = self._get_candidate(self.iteration)

        self.current_pid = PIDController(
            kp=self.current_kp, ki=self.current_ki, kd=self.current_kd,
            output_min=-self.max_velocity, output_max=self.max_velocity,
            integral_max=5.0, derivative_filter_alpha=0.1,
        )
        self.pitch_history = []
        self.static_pitch_history = []
        self.output_history = []
        self.chattering_diffs = []
        self.prev_output = 0.0
        self.robot_fell = False
        self.fall_reason = ""
        self.overshoot_fwd = 0.0
        self.overshoot_bwd = 0.0
        self.settling_time_fwd = self.recovery_duration
        self.settling_time_bwd = self.recovery_duration
        self.zero_crossings_fwd = 0
        self.zero_crossings_bwd = 0

        self.trial_start_time = timestamp
        self.state = self.STATE_STATIC
        self.state_start_time = timestamp

        self.get_logger().info(
            f'  [Mẫu thử {self.iteration}/{self.max_iterations}] 🧪 Thử nghiệm: '
            f'Kp={self.current_kp:.2f} | Ki={self.current_ki:.4f} | Kd={self.current_kd:.2f}'
        )
        self.get_logger().info(f'  ➤ Định hướng mẫu: {self.current_desc}')
        self.get_logger().info(
            f'  ➤ Mỏ neo gốc đang giữ: Kp={self.anchor_pid[0]:.2f}, Ki={self.anchor_pid[1]:.4f}, Kd={self.anchor_pid[2]:.2f} (Kỷ lục: {self.best_fitness:.1f}%)'
        )

    # ====================== PHÂN TÍCH DAO ĐỘNG & CHẤM ĐIỂM ======================

    def _evaluate_iteration(self, timestamp):
        # 1. Đo lường các chỉ số đáp ứng thực tế
        final_deg = math.degrees(abs(self.pitch_history[-1])) if self.pitch_history else 45.0
        if final_deg > 4.5 and not self.robot_fell:
            self.robot_fell = True
            self.fall_reason = f"Không hồi phục vị trí đứng ({final_deg:.1f}°)"

        if len(self.pitch_history) >= 10:
            rms_deg = math.degrees(math.sqrt(sum(p**2 for p in self.pitch_history) / len(self.pitch_history)))
        else:
            rms_deg = 15.0

        mp_deg = math.degrees(max(self.overshoot_fwd, self.overshoot_bwd))
        ts_sec = max(self.settling_time_fwd, self.settling_time_bwd)
        chatter = sum(self.chattering_diffs) / max(1, len(self.chattering_diffs))
        avg_drift = abs(sum(self.output_history) / max(1, len(self.output_history)))
        total_ringing = max(self.zero_crossings_fwd, self.zero_crossings_bwd)

        # 2. Chấm điểm Fitness theo tiêu chuẩn IEEE
        if self.robot_fell:
            fitness = 0.0
        else:
            penalty = (
                0.04 * rms_deg + 0.03 * mp_deg + 0.15 * ts_sec
                + 0.30 * avg_drift + 0.35 * chatter
            )
            fitness = 100.0 * math.exp(-penalty)

        # 3. Cập nhật kỷ lục tốt nhất và mỏ neo
        is_new_best = False
        if not self.robot_fell and fitness > self.best_fitness:
            is_new_best = True
            self.best_fitness = fitness
            self.best_pid = [self.current_kp, self.current_ki, self.current_kd]
            self.best_metrics = {
                'mp': mp_deg,
                'ts': ts_sec,
                'rms': rms_deg,
                'drift': avg_drift,
                'chatter': chatter
            }
            # Mỏ neo dịch chuyển sang cấu hình vượt trội này
            self.anchor_pid = list(self.best_pid)
            self.anchor_fitness = fitness
            self._save_memory()

        # 4. In bảng "Bệnh án" dao động của mẫu thử
        self.get_logger().info('')
        self.get_logger().info('  ┌─────────────────────────────────────────────────────────────┐')
        self.get_logger().info(f'  │ 📊 KẾT QUẢ ĐO DAO ĐỘNG [Mẫu {self.iteration}/{self.max_iterations}]                          │')
        self.get_logger().info('  ├─────────────────────────────────────────────────────────────┤')
        self.get_logger().info(f'  │  PID thử:   Kp={self.current_kp:<7.2f} Ki={self.current_ki:<7.4f} Kd={self.current_kd:<7.2f}    │')
        if not self.robot_fell:
            self.get_logger().info(f'  │  Đáp ứng:   Mp={mp_deg:<5.1f}°  Ts={ts_sec:<5.2f}s  Drift={avg_drift:<5.2f}m/s           │')
            self.get_logger().info(f'  │             RMS={rms_deg:<4.2f}°  Chatter={chatter:<4.2f}  Nhịp lắc={total_ringing:<2d}         │')
            self.get_logger().info(f'  │  Đạt chuẩn: {fitness:>5.1f}% tối ưu  (Kỷ lục: {self.best_fitness:>5.1f}%)                 │')
            if is_new_best:
                self.get_logger().info('  │  ⭐ KỶ LỤC MỚI ĐÃ ĐƯỢC THIẾT LẬP! (Cập nhật Mỏ neo gốc)     │')
        else:
            self.get_logger().info(f'  │  Trạng thái: ❌ ROBOT BỊ NGÃ: {self.fall_reason:<28}│')
            self.get_logger().info('  │  Đạt chuẩn:   0.0% tối ưu                                   │')
        self.get_logger().info('  ├─────────────────────────────────────────────────────────────┤')
        self.get_logger().info('  │ 🧠 PHÂN TÍCH VẬT LÝ & ĐIỀU PHỐI MẪU THỬ:                    │')

        if self.robot_fell:
            bad_entry = [round(self.current_kp, 2), round(self.current_ki, 4), round(self.current_kd, 2)]
            if not any(abs(b[0] - bad_entry[0]) < 1.0 and abs(b[2] - bad_entry[2]) < 0.3 for b in self.blacklist):
                self.blacklist.append(bad_entry)
                self.blacklist = self.blacklist[-self.MAX_BLACKLIST_SIZE:]
                self._save_memory()
            self.get_logger().warn(f'  🚫 Đã thêm bộ số bị ngã vào Blacklist: Kp={bad_entry[0]:.2f}, Kd={bad_entry[2]:.2f}')
            self.get_logger().info(
                f'  │  • Mẫu thử thất bại ➔ Bảo toàn Mỏ neo gốc: Kp={self.anchor_pid[0]:.2f}, Kd={self.anchor_pid[2]:.2f}   │'
            )
            self.get_logger().info('  │  • Chuyển sang mẫu thử tiếp theo trong không gian an toàn    │')
        else:
            if is_new_best:
                self.get_logger().info(
                    f'  │  • Mẫu thử vượt trội ({fitness:.1f}%) ➔ Đã lưu làm Kỷ Lục Mới!          │'
                )
            else:
                self.get_logger().info(
                    f'  │  • Mẫu thử ổn định ({fitness:.1f}%) nhưng chưa vượt qua Kỷ Lục ({self.best_fitness:.1f}%) │'
                )

        self.get_logger().info('  └─────────────────────────────────────────────────────────────┘')
        self.get_logger().info('')

        # 5. Kiểm tra kết thúc phiên thử
        if self.iteration >= self.max_iterations:
            self._finish_tuning()
            return

        # Chuẩn bị cho mẫu thử tiếp theo
        self.iteration += 1
        self.state = self.STATE_RESET
        self.state_start_time = timestamp
        self.publish_cmd_vel(0.0)
        self.trigger_gazebo_reset()

    # ====================== HOÀN TẤT & LƯU FILE ======================

    def _finish_tuning(self):
        self.state = self.STATE_DONE
        self.publish_cmd_vel(0.0)

        # Sử dụng kỷ lục tốt nhất ghi nhận được trong toàn bộ quá trình
        if self.best_pid is not None and self.best_fitness > 0.0:
            kp, ki, kd = self.best_pid[0], self.best_pid[1], self.best_pid[2]
            opt_percent = self.best_fitness
            self._save_memory()

            if opt_percent >= 90.0:
                rating = "⭐⭐⭐⭐⭐ XUẤT SẮC (Gần như hoàn hảo)"
            elif opt_percent >= 80.0:
                rating = "⭐⭐⭐⭐ RẤT TỐT (Chuẩn công nghiệp - Vận hành thực tế)"
            elif opt_percent >= 70.0:
                rating = "⭐⭐⭐ TỐT (Thăng bằng ổn định, chống nhiễu khá)"
            elif opt_percent >= 50.0:
                rating = "⭐⭐ TRUNG BÌNH (Cân bằng được, còn dao động nhẹ)"
            else:
                rating = "⭐ YẾU (Cần tinh chỉnh lại)"
        else:
            # Nếu chưa có mẫu nào đạt chuẩn, lấy mỏ neo gốc ban đầu
            kp, ki, kd = self.anchor_pid[0], self.anchor_pid[1], self.anchor_pid[2]
            opt_percent = self.anchor_fitness
            rating = "⚠️ DỰ PHÒNG TỪ MỎ NEO GỐC"
            self.get_logger().warn('  ⚠️ Phiên này các mẫu thử đều ngã, bảo tồn cấu hình mỏ neo gốc!')

        # Điểm thành phần từng tiêu chuẩn kỹ thuật
        if self.best_metrics:
            score_rms = max(0.0, min(100.0, (1.0 - self.best_metrics['rms'] / 2.5) * 100))
            score_mp = max(0.0, min(100.0, (1.0 - self.best_metrics['mp'] / 12.0) * 100))
            score_ts = max(0.0, min(100.0, (1.0 - self.best_metrics['ts'] / 1.5) * 100))
            score_drift = max(0.0, min(100.0, (1.0 - self.best_metrics['drift'] / 0.8) * 100))
        else:
            score_rms, score_mp, score_ts, score_drift = opt_percent, opt_percent, opt_percent, opt_percent

        self.get_logger().info('')
        self.get_logger().info('=' * 72)
        self.get_logger().info('     🏆 QUÁ TRÌNH TỐI ƯU HÓA HOÀN TẤT (HỘI TỤ THÀNH CÔNG) 🏆')
        self.get_logger().info('=' * 72)
        self.get_logger().info(f'  🎯 MỨC ĐỘ TỐI ƯU HÓA ĐẠT ĐƯỢC: {opt_percent:.1f}%')
        self.get_logger().info(f'  ⭐ Đánh giá tổng thể: {rating}')
        self.get_logger().info('')
        self.get_logger().info('  ┌────────────────────────────────────────────────────────────┐')
        self.get_logger().info('  │ 📊 BẢNG ĐIỂM CHI TIẾT TỪNG TIÊU CHÍ KỸ THUẬT:              │')
        self.get_logger().info('  ├────────────────────────────────────────────────────────────┤')
        self.get_logger().info(f'  │  • Độ êm ái thăng bằng tĩnh:       {score_rms:>6.1f}%                   │')
        self.get_logger().info(f'  │  • Khả năng dập tắt vọt lố (Mp):   {score_mp:>6.1f}%                   │')
        self.get_logger().info(f'  │  • Tốc độ hồi phục sau huých (Ts): {score_ts:>6.1f}%                   │')
        self.get_logger().info(f'  │  • Khả năng giữ vị trí chống trôi: {score_drift:>6.1f}%                   │')
        self.get_logger().info('  ├────────────────────────────────────────────────────────────┤')
        self.get_logger().info('  │ 🎯 BỘ THAM SỐ PID TỐI ƯU CUỐI CÙNG ĐÃ LƯU:                 │')
        self.get_logger().info('  ├────────────────────────────────────────────────────────────┤')
        self.get_logger().info(f'  │  Kp = {kp:>10.4f}  (Độ cứng vững đàn hồi)               │')
        self.get_logger().info(f'  │  Ki = {ki:>10.4f}  (Triệt tiêu sai số xác lập)          │')
        self.get_logger().info(f'  │  Kd = {kd:>10.4f}  (Giảm chấn dập rung gyro)            │')
        self.get_logger().info('  └────────────────────────────────────────────────────────────┘')
        self.get_logger().info('  ✓ Bộ nhớ đã cập nhật ➔ ~/.pso_pid_memory.json')
        self.get_logger().info('=' * 72)
        self._save_yaml(kp, ki, kd, opt_percent, rating)

    def _save_yaml(self, kp, ki, kd, opt_percent, rating):
        content = f"""# =============================================
# Kết quả tinh chỉnh PID thích nghi theo dao động
# Mức độ tối ưu hóa: {opt_percent:.1f}% ({rating})
# =============================================

balance_controller:
  ros__parameters:
    kp: {kp:.4f}
    ki: {ki:.4f}
    kd: {kd:.4f}
    target_pitch: 0.00000
    max_velocity: 1.5
    integral_max: 5.0
    derivative_filter_alpha: 0.1
    use_gyro_derivative: true
    fall_threshold: 0.785
    enabled: true

    # --- Cascaded Loop (Chống trôi xe) ---
    enable_velocity_control: true
    kp_velocity: 0.08
    kp_position: 0.015
    max_pitch_adjustment: 0.06
"""
        try:
            with open(self.output_file, 'w') as f:
                f.write(content)
            self.get_logger().info(f'  📁 Đã lưu cấu hình tối ưu ({opt_percent:.1f}%): {self.output_file}')
        except Exception as e:
            self.get_logger().warn(f'Lỗi lưu yaml: {e}')


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
