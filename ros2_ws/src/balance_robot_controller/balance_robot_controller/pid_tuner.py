"""
PID Auto-Tuner Node - Hệ thống Tune PID 2 pha:
  Pha 1: Relay Feedback (Ziegler-Nichols) → Tìm PID gần đúng (~15 giây)
  Pha 2: PSO Fine-tune 2D (Kp, Kd)      → Tinh chỉnh quanh kết quả Relay (~2 phút)

Đặc tính:
1. RELAY FEEDBACK (Pha 1):
   - Áp dụng relay (bang-bang) controller tạo dao động đều
   - Đo chu kỳ Tu và biên độ Au → tính Ku (Critical Gain)
   - Tự động tính PID theo công thức Ziegler-Nichols

2. PSO FINE-TUNE (Pha 2):
   - Chỉ tìm 2 tham số (Kp, Kd) → hội tụ nhanh gấp nhiều lần 4D
   - Ki cố định = kết quả Ziegler-Nichols
   - Phạm vi tìm kiếm ±30% quanh kết quả Relay
   - Trí nhớ vĩnh viễn (~/.pso_pid_memory.json)
   - Blacklist giới hạn 5 điểm, bán kính nhỏ

3. FITNESS TIẾN BỘ DẦN:
   - Thế hệ 1-2: Đơn giản (thời gian đứng + RMS pitch)
   - Thế hệ 3:   IEEE benchmark đầy đủ (ITAE, Ts, Mp, Chattering, Drift)
"""

import json
import math
import os
import random
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


class Particle:
    """Đại diện cho 1 cá thể PSO mang gen [Kp, Kd] (2D)."""

    def __init__(self, bounds, initial_pos=None):
        self.bounds = bounds
        if initial_pos is not None:
            self.position = [
                max(bounds[i][0], min(bounds[i][1], initial_pos[i]))
                for i in range(len(bounds))
            ]
        else:
            self.position = [random.uniform(b[0], b[1]) for b in bounds]
        self.velocity = [
            random.uniform(-0.1 * (b[1] - b[0]), 0.1 * (b[1] - b[0]))
            for b in bounds
        ]
        self.best_position = list(self.position)
        self.best_fitness = -float('inf')
        self.current_fitness = 0.0

    def update(self, global_best_pos, blacklist=None, w=0.5, c1=1.5, c2=1.5):
        """Cập nhật vận tốc và vị trí."""
        for i in range(len(self.position)):
            r1, r2 = random.random(), random.random()
            v_cog = c1 * r1 * (self.best_position[i] - self.position[i])
            v_soc = c2 * r2 * (global_best_pos[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + v_cog + v_soc

            v_max = (self.bounds[i][1] - self.bounds[i][0]) * 0.25
            self.velocity[i] = max(-v_max, min(v_max, self.velocity[i]))
            self.position[i] += self.velocity[i]
            self.position[i] = max(self.bounds[i][0], min(self.bounds[i][1], self.position[i]))

        # Né tránh vùng xấu (nhẹ nhàng, bán kính nhỏ)
        if blacklist:
            for bad_pos in blacklist:
                dist_sq = sum(
                    ((self.position[k] - bad_pos[k]) / (self.bounds[k][1] - self.bounds[k][0])) ** 2
                    for k in range(min(len(self.position), len(bad_pos)))
                )
                if dist_sq < 0.01:  # r < 0.1
                    for k in range(len(self.position)):
                        self.position[k] = 0.9 * self.position[k] + 0.1 * global_best_pos[k]


class PIDTunerNode(Node):
    """
    ROS 2 Node: Relay Feedback (Ziegler-Nichols) → PSO Fine-tune 2D.
    """

    # ===== CÁC PHA STATE MACHINE =====
    # Pha 1: Relay Feedback
    STATE_RELAY_STABILIZE = 'RELAY_STABILIZE'
    STATE_RELAY_OSCILLATE = 'RELAY_OSCILLATE'
    # Pha 2: PSO Fine-tune
    STATE_PSO_RESET = 'PSO_RESET'
    STATE_PSO_STATIC = 'PSO_STATIC'
    STATE_PSO_DISTURB_FWD = 'PSO_DISTURB_FWD'
    STATE_PSO_RECOVER_FWD = 'PSO_RECOVER_FWD'
    STATE_PSO_DISTURB_BWD = 'PSO_DISTURB_BWD'
    STATE_PSO_RECOVER_BWD = 'PSO_RECOVER_BWD'
    STATE_DONE = 'DONE'

    SAFE_DEFAULT_PID = [58.0, 0.75, 6.5, 0.0]
    MIN_TRUSTWORTHY_FITNESS = 15.0
    MAX_BLACKLIST_SIZE = 5

    def __init__(self):
        super().__init__('pid_tuner')

        # ===== Parameters =====
        self.declare_parameter('fall_threshold', 0.785)
        self.declare_parameter('max_velocity', 1.5)
        self.declare_parameter('relay_amplitude', 0.5)
        self.declare_parameter('relay_duration', 12.0)
        self.declare_parameter('relay_hysteresis', 0.005)
        self.declare_parameter('num_particles', 4)
        self.declare_parameter('max_generations', 3)
        self.declare_parameter('balance_duration', 4.5)
        self.declare_parameter('recovery_duration', 4.5)
        self.declare_parameter('memory_file', '~/.pso_pid_memory.json')
        self.declare_parameter('output_file', '~/tuned_pid_params.yaml')
        self.declare_parameter('reset_memory', False)

        self.fall_threshold = self.get_parameter('fall_threshold').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.relay_amplitude = self.get_parameter('relay_amplitude').value
        self.relay_duration = self.get_parameter('relay_duration').value
        self.relay_hysteresis = self.get_parameter('relay_hysteresis').value
        self.num_particles = self.get_parameter('num_particles').value
        self.max_generations = self.get_parameter('max_generations').value
        self.balance_duration = self.get_parameter('balance_duration').value
        self.recovery_duration = self.get_parameter('recovery_duration').value
        self.memory_file = os.path.expanduser(self.get_parameter('memory_file').value)
        self.output_file = os.path.expanduser(self.get_parameter('output_file').value)
        reset_memory = self.get_parameter('reset_memory').value

        # ===== Bộ nhớ =====
        self.blacklist = []
        self.global_best_position = None
        self.global_best_fitness = 0.0

        if reset_memory:
            self._delete_memory()
            self.get_logger().warn('🗑️  ĐÃ XÓA SẠCH BỘ NHỚ (reset_memory=True)!')
        else:
            self._load_memory()

        # ===== State machine =====
        self.state = self.STATE_RELAY_STABILIZE
        self.state_start_time = None

        # ===== Relay Feedback =====
        self.relay_sign = 1
        self.relay_pitch_peaks = []
        self.relay_last_pitch = 0.0
        self.relay_pitch_rising = True
        self.relay_ku = 0.0
        self.relay_tu = 0.0
        self.zn_kp = self.SAFE_DEFAULT_PID[0]
        self.zn_ki = self.SAFE_DEFAULT_PID[1]
        self.zn_kd = self.SAFE_DEFAULT_PID[2]

        # ===== PSO =====
        self.particles = []
        self.pso_bounds = None
        self.current_generation = 1
        self.current_particle_idx = 0
        self.current_pid = None
        self.pitch_history = []
        self.output_history = []
        self.static_pitch_history = []
        self.chattering_diffs = []
        self.prev_output = 0.0
        self.robot_fell = False
        self.overshoot_fwd = 0.0
        self.overshoot_bwd = 0.0
        self.settling_time_fwd = self.recovery_duration
        self.settling_time_bwd = self.recovery_duration
        self.itae_fwd = 0.0
        self.itae_bwd = 0.0
        self.trial_start_time = None

        # ===== Gazebo =====
        self.reset_world_cli = self.create_client(Empty, '/reset_world')
        self.reset_sim_cli = self.create_client(Empty, '/reset_simulation')

        # ===== ROS Interfaces =====
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, 'pid_tuner/status', 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, qos_profile_sensor_data
        )

        self.get_logger().info('')
        self.get_logger().info('=' * 68)
        self.get_logger().info('  🔬 HỆ THỐNG TUNE PID 2 PHA')
        self.get_logger().info('     Pha 1: Relay Feedback (Ziegler-Nichols)')
        self.get_logger().info('     Pha 2: PSO Fine-tune 2D (Kp, Kd)')
        self.get_logger().info('=' * 68)
        self.get_logger().info(f'  Relay: ±{self.relay_amplitude} m/s, tối đa {self.relay_duration:.0f}s')
        self.get_logger().info(f'  PSO:   {self.num_particles} hạt × {self.max_generations} thế hệ')
        if self.global_best_position and self.global_best_fitness >= self.MIN_TRUSTWORTHY_FITNESS:
            self.get_logger().info(f'  ⭐ Kế thừa kỷ lục: Fitness={self.global_best_fitness:.1f}/100')
        self.get_logger().info('=' * 68)
        self.get_logger().info('')

    # ====================== BỘ NHỚ ======================

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    loaded_fit = data.get('global_best_fitness', 0.0)
                    if loaded_fit >= self.MIN_TRUSTWORTHY_FITNESS:
                        self.global_best_position = data.get('global_best_position')
                        self.global_best_fitness = loaded_fit
                        self.get_logger().info(f'  ✅ Kế thừa kỷ lục: Fitness={loaded_fit:.1f}/100')
                    else:
                        self.get_logger().warn(f'  ⚠️ Kỷ lục cũ tệ ({loaded_fit:.1f}). Bắt đầu lại!')
                    if 'zn_kp' in data:
                        self.zn_kp = data['zn_kp']
                        self.zn_ki = data['zn_ki']
                        self.zn_kd = data['zn_kd']
                    self.blacklist = data.get('blacklist', [])[-self.MAX_BLACKLIST_SIZE:]
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
            'global_best_position': self.global_best_position,
            'global_best_fitness': self.global_best_fitness,
            'blacklist': self.blacklist[-self.MAX_BLACKLIST_SIZE:],
            'zn_kp': self.zn_kp,
            'zn_ki': self.zn_ki,
            'zn_kd': self.zn_kd,
        }
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f'Lỗi lưu memory: {e}')

    # ====================== UTILITIES ======================

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

    def _add_to_blacklist(self, reason):
        if self.current_particle_idx < len(self.particles):
            bad_pos = list(self.particles[self.current_particle_idx].position)
            self.blacklist.append(bad_pos)
            self.blacklist = self.blacklist[-self.MAX_BLACKLIST_SIZE:]
            self._save_memory()
            self.get_logger().warn(f'  ⚠️ #{self.current_particle_idx + 1}: {reason} → Blacklist 🚫')

    # ====================== STATE MACHINE ======================

    def imu_callback(self, msg):
        pitch = quaternion_to_pitch(msg.orientation)
        gyro_y = msg.angular_velocity.y
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.state_start_time is None or timestamp < self.state_start_time:
            self.state_start_time = timestamp
        elapsed = timestamp - self.state_start_time

        # ============ PHA 1: RELAY FEEDBACK ============

        if self.state == self.STATE_RELAY_STABILIZE:
            if abs(pitch) > self.fall_threshold:
                self.publish_cmd_vel(0.0)
                if elapsed > 1.0:
                    self.trigger_gazebo_reset()
                    self.state_start_time = timestamp
                return
            out = 58.0 * pitch + 6.5 * gyro_y
            self.publish_cmd_vel(out)
            if elapsed < 2.0:
                return
            if abs(pitch) < 0.175:
                self.state = self.STATE_RELAY_OSCILLATE
                self.state_start_time = timestamp
                self.relay_pitch_peaks = []
                self.relay_last_pitch = pitch
                self.relay_pitch_rising = True
                self.get_logger().info('')
                self.get_logger().info('  ═══════════════════════════════════════════')
                self.get_logger().info('  📡 PHA 1: RELAY FEEDBACK (Ziegler-Nichols)')
                self.get_logger().info('  ═══════════════════════════════════════════')
                self.get_logger().info(f'  Relay ±{self.relay_amplitude} m/s | Đang tạo dao động...')
                self.get_logger().info('')
            elif elapsed > 5.0:
                self.get_logger().warn('  🔄 Robot chưa đứng! Reset Gazebo...')
                self.publish_cmd_vel(0.0)
                self.trigger_gazebo_reset()
                self.state_start_time = timestamp
            return

        if self.state == self.STATE_RELAY_OSCILLATE:
            if pitch > self.relay_hysteresis:
                self.relay_sign = 1
            elif pitch < -self.relay_hysteresis:
                self.relay_sign = -1

            output = self.relay_amplitude * self.relay_sign
            self.publish_cmd_vel(output)

            currently_rising = pitch > self.relay_last_pitch
            if self.relay_pitch_rising and not currently_rising and elapsed > 0.3:
                self.relay_pitch_peaks.append((timestamp, abs(self.relay_last_pitch), 'peak'))
                self.get_logger().info(
                    f'    📈 Đỉnh #{len(self.relay_pitch_peaks)}: '
                    f'{math.degrees(self.relay_last_pitch):+.2f}°'
                )
            elif not self.relay_pitch_rising and currently_rising and elapsed > 0.3:
                self.relay_pitch_peaks.append((timestamp, abs(self.relay_last_pitch), 'valley'))
                self.get_logger().info(
                    f'    📉 Đáy  #{len(self.relay_pitch_peaks)}: '
                    f'{math.degrees(self.relay_last_pitch):+.2f}°'
                )
            self.relay_pitch_rising = currently_rising
            self.relay_last_pitch = pitch

            if abs(pitch) > self.fall_threshold:
                self.publish_cmd_vel(0.0)
                self.get_logger().warn('  ⚠️ Ngã khi relay! Dùng PID mặc định.')
                self.zn_kp = self.SAFE_DEFAULT_PID[0]
                self.zn_ki = self.SAFE_DEFAULT_PID[1]
                self.zn_kd = self.SAFE_DEFAULT_PID[2]
                self.trigger_gazebo_reset()
                self._transition_to_pso(timestamp)
                return

            n = len(self.relay_pitch_peaks)
            if n >= 6 or (elapsed > self.relay_duration and n >= 4):
                self._calculate_ziegler_nichols()
                self.trigger_gazebo_reset()
                self._transition_to_pso(timestamp)
            elif elapsed > self.relay_duration + 4.0:
                self.get_logger().warn(f'  ⚠️ Chỉ {n} đỉnh/đáy (cần ≥4). Dùng PID mặc định.')
                self.zn_kp = self.SAFE_DEFAULT_PID[0]
                self.zn_ki = self.SAFE_DEFAULT_PID[1]
                self.zn_kd = self.SAFE_DEFAULT_PID[2]
                self.trigger_gazebo_reset()
                self._transition_to_pso(timestamp)
            return

        # ============ PHA 2: PSO FINE-TUNE ============

        if abs(pitch) > self.fall_threshold and self.state not in [self.STATE_PSO_RESET, self.STATE_DONE]:
            self.robot_fell = True
            self.publish_cmd_vel(0.0)
            self._add_to_blacklist(f'NGÃ (Pitch={math.degrees(pitch):.1f}°)')
            self._evaluate_and_next_particle(timestamp)
            return

        if self.state == self.STATE_PSO_RESET:
            if abs(pitch) > self.fall_threshold:
                self.publish_cmd_vel(0.0)
                if elapsed > 1.0:
                    self.trigger_gazebo_reset()
                    self.state_start_time = timestamp
                return
            out = self.zn_kp * pitch + self.zn_kd * gyro_y
            self.publish_cmd_vel(out)
            if elapsed < 2.0:
                return
            if abs(pitch) < 0.175:
                self._start_particle_trial(timestamp)
            elif elapsed > 5.0:
                self.get_logger().warn('  🔄 Robot chưa đứng! Reset...')
                self.trigger_gazebo_reset()
                self.state_start_time = timestamp
            return

        if self.state == self.STATE_PSO_STATIC:
            error = pitch
            output = self.current_pid.compute(error, timestamp, measured_rate=gyro_y)
            self.publish_cmd_vel(output)
            self.pitch_history.append(pitch)
            self.static_pitch_history.append(pitch)
            self.output_history.append(output)
            self.chattering_diffs.append(abs(output - self.prev_output))
            self.prev_output = output

            if elapsed > self.balance_duration:
                self.state = self.STATE_PSO_DISTURB_FWD
                self.state_start_time = timestamp
                self.get_logger().info('    👉 [Huých TIẾN] +0.18 m/s')
            return

        if self.state == self.STATE_PSO_DISTURB_FWD:
            error = pitch
            pid_out = self.current_pid.compute(error, timestamp, measured_rate=gyro_y)
            output = pid_out + 0.18
            self.publish_cmd_vel(output)
            self.pitch_history.append(pitch)
            self.output_history.append(output)
            self.chattering_diffs.append(abs(output - self.prev_output))
            self.prev_output = output
            if elapsed > 0.10:
                self.state = self.STATE_PSO_RECOVER_FWD
                self.state_start_time = timestamp
                self.overshoot_fwd = 0.0
            return

        if self.state == self.STATE_PSO_RECOVER_FWD:
            error = pitch
            output = self.current_pid.compute(error, timestamp, measured_rate=gyro_y)
            self.publish_cmd_vel(output)
            self.pitch_history.append(pitch)
            self.output_history.append(output)
            self.chattering_diffs.append(abs(output - self.prev_output))
            self.prev_output = output
            self.overshoot_fwd = max(self.overshoot_fwd, abs(pitch))
            self.itae_fwd += elapsed * abs(pitch) * 0.01
            if abs(pitch) < 0.026 and self.settling_time_fwd >= self.recovery_duration and elapsed > 0.2:
                self.settling_time_fwd = elapsed
            if elapsed > self.recovery_duration:
                self.state = self.STATE_PSO_DISTURB_BWD
                self.state_start_time = timestamp
                self.get_logger().info('    👉 [Huých LÙI] -0.18 m/s')
            return

        if self.state == self.STATE_PSO_DISTURB_BWD:
            error = pitch
            pid_out = self.current_pid.compute(error, timestamp, measured_rate=gyro_y)
            output = pid_out - 0.18
            self.publish_cmd_vel(output)
            self.pitch_history.append(pitch)
            self.output_history.append(output)
            self.chattering_diffs.append(abs(output - self.prev_output))
            self.prev_output = output
            if elapsed > 0.10:
                self.state = self.STATE_PSO_RECOVER_BWD
                self.state_start_time = timestamp
                self.overshoot_bwd = 0.0
            return

        if self.state == self.STATE_PSO_RECOVER_BWD:
            error = pitch
            output = self.current_pid.compute(error, timestamp, measured_rate=gyro_y)
            self.publish_cmd_vel(output)
            self.pitch_history.append(pitch)
            self.output_history.append(output)
            self.chattering_diffs.append(abs(output - self.prev_output))
            self.prev_output = output
            self.overshoot_bwd = max(self.overshoot_bwd, abs(pitch))
            self.itae_bwd += elapsed * abs(pitch) * 0.01
            if abs(pitch) < 0.026 and self.settling_time_bwd >= self.recovery_duration and elapsed > 0.2:
                self.settling_time_bwd = elapsed
            if elapsed > self.recovery_duration:
                self._evaluate_and_next_particle(timestamp)
            return

    # ====================== RELAY TÍNH TOÁN ======================

    def _calculate_ziegler_nichols(self):
        peaks = self.relay_pitch_peaks
        if len(peaks) < 4:
            self.get_logger().warn('  Không đủ dữ liệu relay!')
            return

        periods = []
        for i in range(2, len(peaks)):
            if peaks[i][2] == peaks[i - 2][2]:
                period = peaks[i][0] - peaks[i - 2][0]
                if 0.1 < period < 5.0:
                    periods.append(period)

        if not periods:
            for i in range(1, len(peaks)):
                half = peaks[i][0] - peaks[i - 1][0]
                if 0.05 < half < 2.5:
                    periods.append(half * 2.0)

        if not periods:
            self.get_logger().warn('  Không tính được Tu!')
            return

        self.relay_tu = sum(periods) / len(periods)
        amplitudes = [p[1] for p in peaks]
        au = sum(amplitudes) / len(amplitudes)

        if au < 0.001:
            self.get_logger().warn('  Biên độ quá nhỏ! Tăng relay_amplitude.')
            return

        self.relay_ku = 4.0 * self.relay_amplitude / (math.pi * au)
        self.zn_kp = max(20.0, min(120.0, 0.6 * self.relay_ku))
        self.zn_ki = max(0.1, min(3.0, 1.2 * self.relay_ku / self.relay_tu))
        self.zn_kd = max(1.0, min(15.0, 0.075 * self.relay_ku * self.relay_tu))

        self.get_logger().info('')
        self.get_logger().info('  ┌─────────────────────────────────────────────┐')
        self.get_logger().info('  │       📊 KẾT QUẢ RELAY FEEDBACK             │')
        self.get_logger().info('  ├─────────────────────────────────────────────┤')
        self.get_logger().info(f'  │  Tu = {self.relay_tu:.4f}s  |  Au = {math.degrees(au):.4f}°          │')
        self.get_logger().info(f'  │  Ku (Critical Gain) = {self.relay_ku:.2f}               │')
        self.get_logger().info('  ├─────────────────────────────────────────────┤')
        self.get_logger().info('  │       🎯 ZIEGLER-NICHOLS PID                │')
        self.get_logger().info('  ├─────────────────────────────────────────────┤')
        self.get_logger().info(f'  │  Kp = {self.zn_kp:.2f}  |  Ki = {self.zn_ki:.4f}  |  Kd = {self.zn_kd:.4f}  │')
        self.get_logger().info('  └─────────────────────────────────────────────┘')
        self.get_logger().info('')

    # ====================== PSO CHUYỂN PHA ======================

    def _transition_to_pso(self, timestamp):
        kp_m = max(self.zn_kp * 0.3, 10.0)
        kd_m = max(self.zn_kd * 0.3, 2.0)
        self.pso_bounds = [
            (max(15.0, self.zn_kp - kp_m), min(130.0, self.zn_kp + kp_m)),
            (max(1.0, self.zn_kd - kd_m), min(18.0, self.zn_kd + kd_m)),
        ]

        self.get_logger().info('  ═══════════════════════════════════════════')
        self.get_logger().info('  🐝 PHA 2: PSO FINE-TUNE 2D (Kp, Kd)')
        self.get_logger().info('  ═══════════════════════════════════════════')
        self.get_logger().info(f'  Kp: [{self.pso_bounds[0][0]:.1f} — {self.pso_bounds[0][1]:.1f}]')
        self.get_logger().info(f'  Kd: [{self.pso_bounds[1][0]:.1f} — {self.pso_bounds[1][1]:.1f}]')
        self.get_logger().info(f'  Ki = {self.zn_ki:.4f} (cố định)')
        self.get_logger().info(f'  {self.num_particles} hạt × {self.max_generations} thế hệ')
        self.get_logger().info('')

        self.particles = []
        zn_seed = [self.zn_kp, self.zn_kd]
        self.particles.append(Particle(self.pso_bounds, initial_pos=zn_seed))

        if (self.global_best_position
                and self.global_best_fitness >= self.MIN_TRUSTWORTHY_FITNESS
                and len(self.global_best_position) >= 2):
            self.particles.append(Particle(self.pso_bounds, initial_pos=self.global_best_position[:2]))
        else:
            safe = [self.SAFE_DEFAULT_PID[0], self.SAFE_DEFAULT_PID[2]]
            self.particles.append(Particle(self.pso_bounds, initial_pos=safe))

        for _ in range(2, self.num_particles):
            self.particles.append(Particle(self.pso_bounds))

        if not self.global_best_position or self.global_best_fitness < self.MIN_TRUSTWORTHY_FITNESS:
            self.global_best_position = list(zn_seed)
            self.global_best_fitness = 0.0

        self.current_generation = 1
        self.current_particle_idx = 0
        self.state = self.STATE_PSO_RESET
        self.state_start_time = timestamp

    # ====================== PSO TRIAL ======================

    def _start_particle_trial(self, timestamp):
        particle = self.particles[self.current_particle_idx]
        kp, kd = particle.position[0], particle.position[1]
        ki = self.zn_ki

        self.current_pid = PIDController(
            kp=kp, ki=ki, kd=kd,
            output_min=-self.max_velocity, output_max=self.max_velocity,
            integral_max=5.0, derivative_filter_alpha=0.1,
        )
        self.pitch_history = []
        self.output_history = []
        self.static_pitch_history = []
        self.chattering_diffs = []
        self.prev_output = 0.0
        self.robot_fell = False
        self.overshoot_fwd = 0.0
        self.overshoot_bwd = 0.0
        self.settling_time_fwd = self.recovery_duration
        self.settling_time_bwd = self.recovery_duration
        self.itae_fwd = 0.0
        self.itae_bwd = 0.0
        self.trial_start_time = timestamp
        self.state = self.STATE_PSO_STATIC
        self.state_start_time = timestamp

        labels = {0: '🎯 SEED ZN', 1: '🛡️ AN TOÀN'}
        label = labels.get(self.current_particle_idx, '🔍 Thăm dò')
        if self.current_particle_idx == 1 and self.global_best_fitness >= self.MIN_TRUSTWORTHY_FITNESS:
            label = '⭐ KỶ LỤC'

        self.get_logger().info(
            f'  [Gen {self.current_generation}/{self.max_generations}] '
            f'#{self.current_particle_idx + 1} [{label}]: '
            f'Kp={kp:.2f} | Ki={ki:.4f} | Kd={kd:.2f}'
        )

    # ====================== PSO ĐÁNH GIÁ ======================

    def _evaluate_and_next_particle(self, timestamp):
        particle = self.particles[self.current_particle_idx]
        final_deg = math.degrees(abs(self.pitch_history[-1])) if self.pitch_history else 45.0

        if self.robot_fell or len(self.pitch_history) < 10 or final_deg > 4.5:
            fitness = 0.0
            if final_deg > 4.5 and not self.robot_fell:
                self._add_to_blacklist(f'KHÔNG HỒI PHỤC ({final_deg:.1f}°)')
            self.get_logger().info(f'    ➜ Điểm: 0.0 / 100')
        else:
            standing = timestamp - self.trial_start_time if self.trial_start_time else 0.0
            total = self.balance_duration + 0.1 + self.recovery_duration + 0.1 + self.recovery_duration

            rms_deg = math.degrees(math.sqrt(sum(p**2 for p in self.pitch_history) / len(self.pitch_history)))
            mp_deg = math.degrees(max(self.overshoot_fwd, self.overshoot_bwd))
            ts_sec = max(self.settling_time_fwd, self.settling_time_bwd)
            itae = self.itae_fwd + self.itae_bwd
            chatter = sum(self.chattering_diffs) / max(1, len(self.chattering_diffs))
            avg_drift = abs(sum(self.output_history) / max(1, len(self.output_history)))

            if self.current_generation < self.max_generations:
                # Fitness ĐƠN GIẢN (thế hệ 1, 2)
                time_score = min(standing / total, 1.0) * 70.0
                rms_score = max(0.0, (1.0 - rms_deg / 5.0)) * 30.0
                drift_penalty = max(0.0, (avg_drift - 0.8) * 15.0)
                fitness = max(5.0, time_score + rms_score - drift_penalty)
            else:
                # Fitness IEEE (thế hệ cuối)
                penalty = (
                    0.04 * rms_deg + 0.02 * mp_deg + 0.12 * ts_sec
                    + 0.03 * itae + 0.35 * avg_drift + 0.40 * chatter
                )
                fitness = 100.0 * math.exp(-penalty)

            if fitness > 0:
                mode = 'ĐƠN GIẢN' if self.current_generation < self.max_generations else 'IEEE'
                self.get_logger().info(
                    f'    📊 [{mode}] RMS={rms_deg:.2f}° | Mp={mp_deg:.1f}° | '
                    f'Ts={ts_sec:.2f}s | Drift={avg_drift:.2f}'
                )
                self.get_logger().info(f'    ➜ ĐIỂM: {fitness:.1f} / 100')
            else:
                self.get_logger().info(f'    ➜ Điểm: 0.0 / 100')

        particle.current_fitness = fitness
        if fitness > particle.best_fitness:
            particle.best_fitness = fitness
            particle.best_position = list(particle.position)
        if fitness > self.global_best_fitness:
            self.global_best_fitness = fitness
            self.global_best_position = list(particle.position)
            self.get_logger().info(f'    ⭐ KỶ LỤC MỚI!')
            self._save_memory()

        self.current_particle_idx += 1
        if self.current_particle_idx >= self.num_particles:
            self.get_logger().info('')
            self.get_logger().info(f'  🏁 Thế hệ #{self.current_generation}: Kỷ lục {self.global_best_fitness:.1f}/100')
            if self.current_generation < self.max_generations:
                self.current_generation += 1
                self.current_particle_idx = 0
                for p in self.particles:
                    p.update(self.global_best_position, self.blacklist)
                self.get_logger().info(f'  🚀 Thế hệ #{self.current_generation}...')
                self.get_logger().info('')
            else:
                self._finish_tuning()
                return

        self.state = self.STATE_PSO_RESET
        self.state_start_time = timestamp
        self.trigger_gazebo_reset()

    # ====================== HOÀN TẤT ======================

    def _finish_tuning(self):
        self.state = self.STATE_DONE
        self.publish_cmd_vel(0.0)
        kp = self.global_best_position[0]
        kd = self.global_best_position[1] if len(self.global_best_position) > 1 else self.zn_kd
        ki = self.zn_ki
        self._save_memory()

        self.get_logger().info('')
        self.get_logger().info('=' * 70)
        self.get_logger().info('     🏆 KẾT QUẢ TỐI ƯU HÓA HOÀN TẤT 🏆')
        self.get_logger().info('=' * 70)
        self.get_logger().info(f'  Fitness tối ưu: {self.global_best_fitness:.1f} / 100')
        self.get_logger().info('')
        self.get_logger().info(f'  ┌──────────────────────────────────────────┐')
        self.get_logger().info(f'  │  Kp = {kp:>10.4f}                        │')
        self.get_logger().info(f'  │  Ki = {ki:>10.4f}  (ZN, cố định)         │')
        self.get_logger().info(f'  │  Kd = {kd:>10.4f}                        │')
        self.get_logger().info(f'  └──────────────────────────────────────────┘')
        self.get_logger().info(f'  Relay: Ku={self.relay_ku:.2f} | Tu={self.relay_tu:.4f}s')
        self.get_logger().info('  ✓ Bộ nhớ → ~/.pso_pid_memory.json')
        self.get_logger().info('=' * 70)
        self._save_yaml(kp, ki, kd)

    def _save_yaml(self, kp, ki, kd):
        content = f"""# =============================================
# PID Results: Relay Feedback + PSO Fine-tune 2D
# Fitness: {self.global_best_fitness:.1f}/100
# Relay: Ku={self.relay_ku:.2f}, Tu={self.relay_tu:.4f}s
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
            self.get_logger().info(f'  📁 Đã lưu: {self.output_file}')
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
