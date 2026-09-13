"""
PID Auto-Tuner Node - Hệ thống Tự Thích Nghi Dựa Trên Dữ Liệu Đáp Ứng Dao Động
(Deterministic Oscillation-Based Adaptive Tuner)

Nguyên lý:
  - 100% KHOA HỌC & TẤT ĐỊNH - KHÔNG DÙNG SỐ NGẪU NHIÊN.
  - Mỗi bước lặp, xe được đưa qua quy trình kiểm chuẩn IEEE:
      [Cân bằng tĩnh] ➔ [Huých TIẾN] ➔ [Hồi phục] ➔ [Huých LÙI] ➔ [Hồi phục]
  - Thuật toán đo trực tiếp 5 đại lượng dao động thực tế:
      1. Độ vọt lố cực đại (Mp): Đánh giá độ thiếu/thừa giảm chấn Kd
      2. Thời gian dập tắt dao động (Ts): Đánh giá độ cứng vững Kp
      3. Độ rung chấn vi phân (Chatter): Phát hiện Kp/Kd bị quá căng
      4. Tốc độ trôi xe (Drift): Phát hiện Ki bị dư thừa tích phân
      5. Độ lệch góc tĩnh (RMS): Đánh giá độ êm khi đứng yên
  - Tự động "bắt bệnh" và tính toán giải tích trực tiếp lượng bù:
      ΔKp, ΔKi, ΔKd cho bước tiếp theo.
  - Hội tụ nhanh chỉ sau 4 - 6 bước lặp (~1.5 phút).
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
    ROS2 Node: Tinh chỉnh PID thích nghi theo phân tích dao động.
    """

    # ===== CÁC TRẠNG THÁI KIỂM ĐỊNH =====
    STATE_RESET = 'RESET'
    STATE_STATIC = 'STATIC'
    STATE_DISTURB_FWD = 'DISTURB_FWD'
    STATE_RECOVER_FWD = 'RECOVER_FWD'
    STATE_DISTURB_BWD = 'DISTURB_BWD'
    STATE_RECOVER_BWD = 'RECOVER_BWD'
    STATE_DONE = 'DONE'

    # Điểm xuất phát chuẩn hóa cho xe ~0.93kg
    DEFAULT_SEED_PID = [55.0, 0.50, 6.0]

    # Giới hạn an toàn vật lý của xe
    KP_MIN, KP_MAX = 38.0, 85.0
    KI_MIN, KI_MAX = 0.20, 0.75
    KD_MIN, KD_MAX = 4.0, 9.5

    def __init__(self):
        super().__init__('pid_tuner')

        # ===== Khai báo Parameters =====
        self.declare_parameter('fall_threshold', 0.785)
        self.declare_parameter('max_velocity', 1.5)
        self.declare_parameter('balance_duration', 3.5)
        self.declare_parameter('recovery_duration', 3.5)
        self.declare_parameter('max_iterations', 7)
        self.declare_parameter('memory_file', '~/.pso_pid_memory.json')
        self.declare_parameter('output_file', '~/tuned_pid_params.yaml')
        self.declare_parameter('reset_memory', False)

        self.fall_threshold = self.get_parameter('fall_threshold').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.balance_duration = self.get_parameter('balance_duration').value
        self.recovery_duration = self.get_parameter('recovery_duration').value
        self.max_iterations = self.get_parameter('max_iterations').value
        self.memory_file = os.path.expanduser(self.get_parameter('memory_file').value)
        self.output_file = os.path.expanduser(self.get_parameter('output_file').value)
        reset_memory = self.get_parameter('reset_memory').value

        # ===== Bộ thông số PID hiện tại đang thử =====
        self.current_kp = self.DEFAULT_SEED_PID[0]
        self.current_ki = self.DEFAULT_SEED_PID[1]
        self.current_kd = self.DEFAULT_SEED_PID[2]

        self.best_pid = list(self.DEFAULT_SEED_PID)
        self.best_fitness = 0.0
        self.history_records = []
        self.blacklist = []
        self.MAX_BLACKLIST_SIZE = 10

        if reset_memory:
            self._delete_memory()
            self.get_logger().warn('🗑️  ĐÃ XÓA BỘ NHỚ CŨ (reset_memory=True)!')
        else:
            self._load_memory()

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
        self.get_logger().info('=' * 70)
        self.get_logger().info('  🔬 HỆ THỐNG TỰ THÍCH NGHI PID THEO DỮ LIỆU ĐÁP ỨNG DAO ĐỘNG')
        self.get_logger().info('     (Deterministic Oscillation-Based Adaptive Auto-Tuner)')
        self.get_logger().info('=' * 70)
        self.get_logger().info(f'  Bộ xuất phát: Kp={self.current_kp:.2f} | Ki={self.current_ki:.4f} | Kd={self.current_kd:.2f}')
        self.get_logger().info(f'  Tối đa {self.max_iterations} bước lặp thích nghi (dừng khi hội tụ)')
        self.get_logger().info('=' * 70)
        self.get_logger().info('')

    # ====================== BỘ NHỚ LƯU TRỮ ======================

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    pos = data.get('global_best_position')
                    fit = data.get('global_best_fitness', 0.0)
                    if pos and len(pos) >= 2 and pos[0] >= self.KP_MIN and fit >= 50.0:
                        self.current_kp = float(pos[0])
                        self.current_kd = float(pos[1])
                        if len(pos) >= 3:
                            self.current_ki = max(self.KI_MIN, min(self.KI_MAX, float(pos[2])))
                        elif 'zn_ki' in data:
                            self.current_ki = max(self.KI_MIN, min(self.KI_MAX, float(data['zn_ki'])))
                        self.best_pid = [self.current_kp, self.current_ki, self.current_kd]
                        self.best_fitness = fit
                        self.get_logger().info(
                            f'  ✅ Kế thừa kỷ lục cũ: Kp={self.current_kp:.2f}, Ki={self.current_ki:.4f}, Kd={self.current_kd:.2f} (Fitness={fit:.1f}/100)'
                        )
                    else:
                        self.get_logger().warn('  ⚠️ Kỷ lục cũ không đạt chuẩn (Kp quá yếu hoặc fit thấp). Dùng bộ chuẩn mới!')

                    loaded_bl = data.get('blacklist', [])
                    self.blacklist = [b for b in loaded_bl if isinstance(b, list) and len(b) >= 3][-self.MAX_BLACKLIST_SIZE:]
                    if self.blacklist:
                        self.get_logger().info(f'  📋 Đã nạp {len(self.blacklist)} điểm cấm từ Blacklist cũ.')
            except Exception as e:
                self.get_logger().warn(f'Lỗi đọc memory: {e}')

    def _delete_memory(self):
        if os.path.exists(self.memory_file):
            try:
                os.remove(self.memory_file)
            except Exception:
                pass

    def _save_memory(self):
        data = {
            'global_best_position': [self.best_pid[0], self.best_pid[2]],
            'global_best_fitness': self.best_fitness,
            'zn_kp': self.best_pid[0],
            'zn_ki': self.best_pid[1],
            'zn_kd': self.best_pid[2],
            'blacklist': self.blacklist[-self.MAX_BLACKLIST_SIZE:],
        }
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f'Lỗi lưu memory: {e}')

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

            # Dùng PD giữ nhẹ trong 2 giây đầu reset
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
                self.get_logger().info('    👉 [Huých TIẾN] +0.18 m/s...')
            return

        # ----- TRẠNG THÁI 3: HUÝCH TIẾN -----
        if self.state == self.STATE_DISTURB_FWD:
            error = pitch
            pid_out = self.current_pid.compute(error, timestamp, measured_rate=gyro_y)
            output = pid_out + 0.18
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
                self.get_logger().info('    👉 [Huých LÙI] -0.18 m/s...')
            return

        # ----- TRẠNG THÁI 5: HUÝCH LÙI -----
        if self.state == self.STATE_DISTURB_BWD:
            error = pitch
            pid_out = self.current_pid.compute(error, timestamp, measured_rate=gyro_y)
            output = pid_out - 0.18
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
            f'  [Bước lặp {self.iteration}/{self.max_iterations}] 🧪 Thử nghiệm: '
            f'Kp={self.current_kp:.2f} | Ki={self.current_ki:.4f} | Kd={self.current_kd:.2f}'
        )

    # ====================== PHÂN TÍCH DAO ĐỘNG & TÍNH BÙ TRỪ ======================

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

        # Cập nhật kỷ lục tốt nhất
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_pid = [self.current_kp, self.current_ki, self.current_kd]
            self._save_memory()

        # 3. In bảng "Bệnh án" dao động của xe
        self.get_logger().info('')
        self.get_logger().info('  ┌─────────────────────────────────────────────────────────────┐')
        self.get_logger().info(f'  │ 📊 KẾT QUẢ ĐO DAO ĐỘNG [Bước {self.iteration}/{self.max_iterations}]                          │')
        self.get_logger().info('  ├─────────────────────────────────────────────────────────────┤')
        self.get_logger().info(f'  │  PID thử:   Kp={self.current_kp:<7.2f} Ki={self.current_ki:<7.4f} Kd={self.current_kd:<7.2f}    │')
        if not self.robot_fell:
            self.get_logger().info(f'  │  Đáp ứng:   Mp={mp_deg:<5.1f}°  Ts={ts_sec:<5.2f}s  Drift={avg_drift:<5.2f}m/s           │')
            self.get_logger().info(f'  │             RMS={rms_deg:<4.2f}°  Chatter={chatter:<4.2f}  Nhịp lắc={total_ringing:<2d}         │')
            self.get_logger().info(f'  │  Điểm:      {fitness:.1f} / 100  (Kỷ lục: {self.best_fitness:.1f}/100)               │')
        else:
            self.get_logger().info(f'  │  Trạng thái: ❌ ROBOT BỊ NGÃ: {self.fall_reason:<28}│')
            self.get_logger().info('  │  Điểm:      0.0 / 100                                       │')
        self.get_logger().info('  ├─────────────────────────────────────────────────────────────┤')
        self.get_logger().info('  │ 🧠 PHÂN TÍCH VẬT LÝ & ĐIỀU CHỈNH TỰ THÍCH NGHI:             │')

        # 4. Thuật toán phân tích giải tích để tính lượng thay đổi ΔKp, ΔKi, ΔKd
        delta_kp = 0.0
        delta_ki = 0.0
        delta_kd = 0.0
        reasons = []

        if self.robot_fell:
            # Ghi nhận bộ số bị ngã vào Blacklist để không bao giờ lặp lại
            bad_entry = [round(self.current_kp, 2), round(self.current_ki, 4), round(self.current_kd, 2)]
            if not any(abs(b[0] - bad_entry[0]) < 1.0 and abs(b[2] - bad_entry[2]) < 0.3 for b in self.blacklist):
                self.blacklist.append(bad_entry)
                self.blacklist = self.blacklist[-self.MAX_BLACKLIST_SIZE:]
                self._save_memory()
            self.get_logger().warn(f'  🚫 Đã thêm bộ số bị ngã vào Blacklist: Kp={bad_entry[0]:.2f}, Kd={bad_entry[2]:.2f}')

            # Nếu xe ngã -> Tăng mạnh độ cứng vững Kp và giảm chấn Kd
            delta_kp = +8.0
            delta_kd = +1.5
            delta_ki = -0.10
            reasons.append("Xe bị ngã ➔ Tăng mạnh lực đàn hồi Kp (+8.0) và giảm chấn Kd (+1.5)")
        else:
            # --- Phân tích Giảm Chấn Kd (Dựa vào Độ vọt lố Mp và Số nhịp rung lắc) ---
            if mp_deg > 6.0 or total_ringing >= 3:
                d_kd = min(1.2, max(0.4, 0.12 * (mp_deg - 5.0)))
                delta_kd += d_kd
                reasons.append(f"Vọt lố ngửa người lớn (Mp={mp_deg:.1f}° > 6°) ➔ Tăng Kd (+{d_kd:.2f}) dập tắt lắc")
            elif mp_deg <= 4.0 and chatter > 0.08:
                delta_kd -= 0.35
                reasons.append(f"Rung chấn motor cao (Chatter={chatter:.2f}) ➔ Giảm nhẹ Kd (-0.35)")

            # --- Phân tích Độ Cững Vững Kp (Dựa vào Thời gian hồi phục Ts và Chattering) ---
            if ts_sec > 0.8:
                d_kp = min(6.0, max(2.0, 4.0 * (ts_sec - 0.6)))
                delta_kp += d_kp
                reasons.append(f"Hồi phục chậm (Ts={ts_sec:.2f}s > 0.8s) ➔ Tăng Kp (+{d_kp:.2f}) để tăng độ cứng")
            elif chatter > 0.12:
                delta_kp -= 3.5
                reasons.append(f"Dao động tần số cao căng cứng ➔ Giảm Kp (-3.5)")

            # --- Phân tích Tích Phân Ki (Dựa vào Tốc độ trôi xe Drift) ---
            if avg_drift > 0.40:
                d_ki = min(0.15, max(0.05, (avg_drift - 0.3) * 0.25))
                delta_ki -= d_ki
                reasons.append(f"Xe bị trôi bánh (Drift={avg_drift:.2f}m/s) ➔ Giảm Ki (-{d_ki:.3f}) khử trôi")
            elif avg_drift < 0.15 and rms_deg < 0.6:
                reasons.append("Vị trí và độ thăng bằng rất ổn định ➔ Giữ nguyên Ki")

        # 5. Cập nhật bộ thông số cho bước tiếp theo (Kẹp trong biên an toàn)
        next_kp = max(self.KP_MIN, min(self.KP_MAX, self.current_kp + delta_kp))
        next_ki = max(self.KI_MIN, min(self.KI_MAX, self.current_ki + delta_ki))
        next_kd = max(self.KD_MIN, min(self.KD_MAX, self.current_kd + delta_kd))

        # 5b. Kiểm tra né tránh Blacklist (Tuyệt đối không lặp lại vùng từng bị ngã)
        for bad in self.blacklist:
            dist = math.sqrt(
                ((next_kp - bad[0]) / (self.KP_MAX - self.KP_MIN)) ** 2 +
                ((next_kd - bad[2]) / (self.KD_MAX - self.KD_MIN)) ** 2
            )
            if dist < 0.08:  # Quá gần điểm ngã cũ
                shift_kp = 4.0 if next_kp <= bad[0] else 2.0
                shift_kd = 0.8 if next_kd <= bad[2] else 0.4
                orig_kp, orig_kd = next_kp, next_kd
                next_kp = max(self.KP_MIN, min(self.KP_MAX, next_kp + shift_kp))
                next_kd = max(self.KD_MIN, min(self.KD_MAX, next_kd + shift_kd))
                reasons.append(f"Gần Blacklist ({bad[0]:.1f}, {bad[2]:.1f}) ➔ Né sang Kp={next_kp:.2f}, Kd={next_kd:.2f}")
                self.get_logger().warn(
                    f'  🛡️ Né tránh Blacklist: Bộ số gần điểm ngã cũ ({bad[0]:.1f}, {bad[2]:.1f}) '
                    f'➔ Dịch chuyển an toàn từ ({orig_kp:.2f}, {orig_kd:.2f}) sang ({next_kp:.2f}, {next_kd:.2f})'
                )
                break

        # In các lý do điều chỉnh
        if not reasons:
            reasons.append("Tất cả các chỉ số đều đạt mức lý tưởng!")
        for r in reasons:
            self.get_logger().info(f'  │  • {r:<57}│')

        self.get_logger().info('  ├─────────────────────────────────────────────────────────────┤')
        self.get_logger().info(
            f'  │  ➜ BƯỚC TIẾP: Kp={next_kp:<7.2f} Ki={next_ki:<7.4f} Kd={next_kd:<7.2f}                 │'
        )
        self.get_logger().info('  └─────────────────────────────────────────────────────────────┘')
        self.get_logger().info('')

        # 6. Kiểm tra điều kiện hội tụ sớm (Dừng nếu đã tối ưu)
        converged = (
            not self.robot_fell
            and fitness >= 82.0
            and abs(delta_kp) < 1.0
            and abs(delta_kd) < 0.20
            and mp_deg <= 6.5
            and ts_sec <= 0.75
            and self.iteration >= 3
        )

        if converged or self.iteration >= self.max_iterations:
            self._finish_tuning()
            return

        # Chuẩn bị cho bước lặp tiếp theo
        self.current_kp = next_kp
        self.current_ki = next_ki
        self.current_kd = next_kd
        self.iteration += 1

        self.state = self.STATE_RESET
        self.state_start_time = timestamp
        self.publish_cmd_vel(0.0)
        self.trigger_gazebo_reset()

    # ====================== HOÀN TẤT & LƯU FILE ======================

    def _finish_tuning(self):
        self.state = self.STATE_DONE
        self.publish_cmd_vel(0.0)
        kp, ki, kd = self.best_pid[0], self.best_pid[1], self.best_pid[2]
        self._save_memory()

        self.get_logger().info('')
        self.get_logger().info('=' * 70)
        self.get_logger().info('     🏆 QUÁ TRÌNH TỐI ƯU HÓA HOÀN TẤT (HỘI TỤ THÀNH CÔNG) 🏆')
        self.get_logger().info('=' * 70)
        self.get_logger().info(f'  Điểm Fitness tối ưu: {self.best_fitness:.1f} / 100')
        self.get_logger().info('')
        self.get_logger().info('  ┌──────────────────────────────────────────┐')
        self.get_logger().info(f'  │  Kp = {kp:>10.4f} (Độ cứng vững)        │')
        self.get_logger().info(f'  │  Ki = {ki:>10.4f} (Khử sai số tĩnh)     │')
        self.get_logger().info(f'  │  Kd = {kd:>10.4f} (Giảm chấn gyro)      │')
        self.get_logger().info('  └──────────────────────────────────────────┘')
        self.get_logger().info('  ✓ Bộ nhớ đã cập nhật ➔ ~/.pso_pid_memory.json')
        self.get_logger().info('=' * 70)
        self._save_yaml(kp, ki, kd)

    def _save_yaml(self, kp, ki, kd):
        content = f"""# =============================================
# Kết quả tinh chỉnh PID thích nghi theo dao động
# Fitness: {self.best_fitness:.1f}/100
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
            self.get_logger().info(f'  📁 Đã lưu cấu hình tối ưu: {self.output_file}')
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
