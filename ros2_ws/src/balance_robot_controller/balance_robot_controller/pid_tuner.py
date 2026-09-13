"""
PID Auto-Tuner Node - Sử dụng Thuật toán Tối ưu hóa Bầy đàn (PSO) với Trí nhớ Học liên tục (Persistent Memory).
Tự động tìm bộ PID và Target Pitch tối ưu nhất cho xe cân bằng hai bánh trong Gazebo.

Đặc tính AI nâng cao:
1. TRÍ NHỚ VĨNH VIỄN (~/.pso_pid_memory.json):
   - Lưu trữ kỷ lục tốt nhất từ các lần chạy trước, kế thừa làm hạt giống (Warm Start).
   - "Danh sách đen" (Blacklist / Tabu List): Ghi nhớ các vùng thông số từng làm xe bị ngã hoặc điểm thấp,
     các hạt đời sau sẽ TỰ ĐỘNG NÉ TRÁNH các vùng này.
2. THỬ NGHIỆM KHÁNG LỰC ĐẨY (Disturbance Rejection): Tự động phát xung lực huých xe để đo phản xạ kéo lại.
3. CHUẨN HÓA ĐIỂM FITNESS (Thang 100%): Phản ánh trực quan độ vững chãi của robot.
4. TỰ ĐỘNG RESET GAZEBO (/reset_world) giữa các cá thể.
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
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, String
from std_srvs.srv import Empty

from balance_robot_controller.pid import PIDController


def quaternion_to_pitch(q):
    """Trích xuất góc pitch từ quaternion."""
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    if abs(sinp) >= 1.0:
        return math.copysign(math.pi / 2.0, sinp)
    return math.asin(sinp)


class Particle:
    """Đại diện cho 1 cá thể trong bầy đàn mang bộ gen PID."""

    def __init__(self, bounds, initial_pos=None):
        self.bounds = bounds  # [(min, max) cho Kp, Ki, Kd, Target_Pitch]

        if initial_pos is not None:
            self.position = list(initial_pos)
        else:
            self.position = [random.uniform(b[0], b[1]) for b in bounds]

        # Vận tốc bay ban đầu
        self.velocity = [
            random.uniform(-0.1 * (b[1] - b[0]), 0.1 * (b[1] - b[0]))
            for b in bounds
        ]
        self.best_position = list(self.position)
        self.best_fitness = -float('inf')
        self.current_fitness = 0.0

    def update(self, global_best_pos, blacklist=None, w=0.5, c1=1.5, c2=1.5):
        """Cập nhật vận tốc và vị trí, chủ động né tránh các vùng trong blacklist."""
        for i in range(len(self.position)):
            r1 = random.random()
            r2 = random.random()

            # Quán tính + Hướng về Best cá nhân + Hướng về Best bầy đàn
            v_cog = c1 * r1 * (self.best_position[i] - self.position[i])
            v_soc = c2 * r2 * (global_best_pos[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + v_cog + v_soc

            # Kẹp vận tốc tối đa
            v_max = (self.bounds[i][1] - self.bounds[i][0]) * 0.25
            self.velocity[i] = max(-v_max, min(v_max, self.velocity[i]))

            # Cập nhật vị trí
            self.position[i] += self.velocity[i]
            self.position[i] = max(self.bounds[i][0], min(self.bounds[i][1], self.position[i]))

        # Cơ chế Né tránh Vùng Xấu (Tabu Avoidance) - nhẹ nhàng
        if blacklist:
            for bad_pos in blacklist:
                # Tính khoảng cách chuẩn hóa tới điểm xấu
                dist_sq = sum(
                    ((self.position[k] - bad_pos[k]) / (self.bounds[k][1] - self.bounds[k][0])) ** 2
                    for k in range(len(self.position))
                )
                if dist_sq < 0.01:  # Bán kính nguy hiểm thu nhỏ (r < 0.1)
                    # Đẩy nhẹ hạt ra xa vùng xấu (giữ 90% vị trí hiện tại)
                    for k in range(len(self.position)):
                        self.position[k] = 0.9 * self.position[k] + 0.1 * global_best_pos[k]


class PsoPIDTunerNode(Node):
    """
    ROS 2 Node thực thi thuật toán PSO có Trí nhớ vĩnh viễn (Memory Persistence).
    """

    # ===== CÁC PHA KIỂM ĐỊNH ĐIỀU KHIỂN HỌC (IEEE CONTROL BENCHMARK) =====
    STATE_RESET = 'RESET'
    STATE_STATIC = 'STATIC'             # 1. Kiểm định độ ổn định tĩnh & độ êm (2.5s)
    STATE_DISTURB_FWD = 'DISTURB_FWD'   # 2. Xung huých TIẾN (+0.18 m/s, 0.10s)
    STATE_RECOVER_FWD = 'RECOVER_FWD'   # 3. Đo hồi phục TIẾN: Ts, Mp, ITAE (3.0s)
    STATE_DISTURB_BWD = 'DISTURB_BWD'   # 4. Xung huých LÙI (-0.18 m/s, 0.10s)
    STATE_RECOVER_BWD = 'RECOVER_BWD'   # 5. Đo hồi phục LÙI: Ts, Mp, ITAE (3.0s)
    STATE_DONE = 'DONE'

    # Không gian tìm kiếm 4 chiều: [Kp, Ki, Kd, Target_Pitch]
    SEARCH_BOUNDS = [
        (40.0, 85.0),       # Kp (khoảng rộng tối ưu: 40 phản xạ êm -> 85 phản xạ đanh)
        (0.2, 1.2),         # Ki (nhỏ để chống trôi)
        (4.5, 9.0),         # Kd
        (-0.008, 0.008),    # Target Pitch (rad)
    ]

    # Bộ PID an toàn mặc định (đã biết giữ xe đứng trong Gazebo)
    SAFE_DEFAULT_PID = [58.0, 0.75, 6.5, 0.0]
    # Ngưỡng fitness tối thiểu để coi là "đáng tin cậy" kế thừa
    MIN_TRUSTWORTHY_FITNESS = 15.0
    # Số điểm blacklist tối đa (ít để tránh bịt kín không gian tìm kiếm)
    MAX_BLACKLIST_SIZE = 5

    def __init__(self):
        super().__init__('pid_tuner')

        # ===== Parameters =====
        self.declare_parameter('num_particles', 6)
        self.declare_parameter('max_generations', 3)
        self.declare_parameter('fall_threshold', 0.785)
        self.declare_parameter('max_velocity', 1.5)
        self.declare_parameter('balance_duration', 4.5)     # Thời gian thử đứng yên (tăng lên 4.5s)
        self.declare_parameter('recovery_duration', 4.5)    # Thời gian thử sau huých (tăng lên 4.5s)
        self.declare_parameter('memory_file', '~/.pso_pid_memory.json')
        self.declare_parameter('output_file', '~/tuned_pid_params.yaml')
        self.declare_parameter('reset_memory', False)       # Đặt True để xóa sạch bộ nhớ, bắt đầu lại từ đầu

        self.num_particles = self.get_parameter('num_particles').value
        self.max_generations = self.get_parameter('max_generations').value
        self.fall_threshold = self.get_parameter('fall_threshold').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.balance_duration = self.get_parameter('balance_duration').value
        self.recovery_duration = self.get_parameter('recovery_duration').value
        self.memory_file = os.path.expanduser(self.get_parameter('memory_file').value)
        self.output_file = os.path.expanduser(self.get_parameter('output_file').value)
        reset_memory = self.get_parameter('reset_memory').value

        # Khởi tạo hoặc nạp bộ nhớ AI từ file
        self.blacklist = []
        self.history_records = []
        self.global_best_position = list(self.SAFE_DEFAULT_PID)
        self.global_best_fitness = 0.0

        if reset_memory:
            # Người dùng yêu cầu xóa sạch bộ nhớ cũ
            self._delete_memory()
            self.get_logger().warn('🗑️  ĐÃ XÓA SẠCH BỘ NHỚ HỌC CŨ (reset_memory=True). Bắt đầu từ đầu!')
        else:
            self._load_memory()

        # Khởi tạo bầy đàn
        self.particles = []
        # Hạt #1 luôn là Hạt giống Kỷ lục Tốt nhất (Seed Champion)
        self.particles.append(Particle(self.SEARCH_BOUNDS, initial_pos=self.global_best_position))
        # Hạt #2 luôn là Hạt An toàn Mặc định (Safety Net - tránh mất gốc)
        self.particles.append(Particle(self.SEARCH_BOUNDS, initial_pos=self.SAFE_DEFAULT_PID))

        # Các hạt còn lại phân bố tìm kiếm xung quanh
        for _ in range(2, self.num_particles):
            p = Particle(self.SEARCH_BOUNDS)
            # Khởi tạo né tránh blacklist ngay từ đầu
            p.update(self.global_best_position, self.blacklist)
            self.particles.append(p)

        # Quản lý tiến trình
        self.current_generation = 1
        self.current_particle_idx = 0
        self.state = self.STATE_RESET
        self.state_start_time = None

        # Dữ liệu lượt thử
        self.current_pid = None
        self.current_target_pitch = 0.0
        self.pitch_history = []
        self.output_history = []
        self.max_recovery_pitch = 0.0
        self.robot_fell = False

        # Service Gazebo Reset
        self.reset_world_cli = self.create_client(Empty, '/reset_world')
        self.reset_sim_cli = self.create_client(Empty, '/reset_simulation')

        # Publishers / Subscriber
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, 'pid_tuner/status', 10)
        self.data_pub = self.create_publisher(Float64MultiArray, 'pid_tuner/data', 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, qos_profile_sensor_data
        )

        self.get_logger().info('')
        self.get_logger().info('=' * 68)
        self.get_logger().info('  🧠 THUẬT TOÁN BẦY ĐÀN (PSO) VỚI BỘ NHỚ HỌC LIÊN TỤC (MEMORY) 🐝')
        self.get_logger().info('=' * 68)
        if self.global_best_fitness > 0:
            self.get_logger().info(
                f'  ⭐ Kế thừa kỷ lục cũ: Kp={self.global_best_position[0]:.2f}, '
                f'Kd={self.global_best_position[2]:.2f} (Fitness: {self.global_best_fitness:.1f}/100)'
            )
            self.get_logger().info(f'  🚫 Vùng cấm (Blacklist) đã ghi nhớ: {len(self.blacklist)} điểm xấu (Sẽ tránh xa)')
        else:
            self.get_logger().info('  Khởi tạo bộ nhớ tìm kiếm ban đầu...')
        self.get_logger().info(f'  Quy mô: {self.num_particles} hạt x {self.max_generations} thế hệ')
        self.get_logger().info('=' * 68)
        self.get_logger().info('')

    def _load_memory(self):
        """Nạp dữ liệu học từ file, tự động khử nhiễm nếu bộ nhớ bị hỏng."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    if 'global_best_position' in data:
                        loaded_pos = data['global_best_position']
                        loaded_fit = data.get('global_best_fitness', 0.0)

                        # ===== KHỬ NHIỄM: Nếu best cũ quá tệ → bỏ, dùng mặc định an toàn =====
                        if loaded_fit >= self.MIN_TRUSTWORTHY_FITNESS:
                            self.global_best_position = loaded_pos
                            self.global_best_fitness = loaded_fit
                            self.get_logger().info(
                                f'  ✅ Kế thừa kỷ lục cũ đáng tin cậy: Fitness={loaded_fit:.1f}/100'
                            )
                        else:
                            self.get_logger().warn(
                                f'  ⚠️ Kỷ lục cũ quá tệ (Fitness={loaded_fit:.1f} < {self.MIN_TRUSTWORTHY_FITNESS}). '
                                f'ĐÃ BỎ QUA, dùng PID an toàn mặc định [Kp=58, Kd=6.5]!'
                            )
                            # Giữ nguyên SAFE_DEFAULT_PID, không load cái cũ

                    # Blacklist: chỉ giữ tối đa MAX_BLACKLIST_SIZE điểm và loại trùng lặp
                    raw_blacklist = data.get('blacklist', [])
                    self.blacklist = self._deduplicate_blacklist(raw_blacklist)
            except Exception as e:
                self.get_logger().warn(f'Không thể đọc file memory: {e}')

    def _deduplicate_blacklist(self, blacklist):
        """Loại bỏ các điểm blacklist quá gần nhau (trùng lặp) và giới hạn kích thước."""
        if not blacklist:
            return []
        cleaned = []
        for point in blacklist:
            is_duplicate = False
            for existing in cleaned:
                # Khoảng cách chuẩn hóa giữa 2 điểm
                dist_sq = sum(
                    ((point[k] - existing[k]) / (self.SEARCH_BOUNDS[k][1] - self.SEARCH_BOUNDS[k][0])) ** 2
                    for k in range(len(point))
                )
                if dist_sq < 0.01:  # Quá gần nhau → coi như trùng
                    is_duplicate = True
                    break
            if not is_duplicate:
                cleaned.append(point)
        # Chỉ giữ N điểm gần nhất (mới nhất)
        return cleaned[-self.MAX_BLACKLIST_SIZE:]

    def _delete_memory(self):
        """Xóa sạch file bộ nhớ để bắt đầu lại từ đầu."""
        if os.path.exists(self.memory_file):
            try:
                os.remove(self.memory_file)
            except Exception as e:
                self.get_logger().warn(f'Không thể xóa file memory: {e}')

    def _save_memory(self):
        """Lưu lại trí nhớ học được vào ổ đĩa."""
        # Loại trùng và giới hạn kích thước trước khi lưu
        self.blacklist = self._deduplicate_blacklist(self.blacklist)
        data = {
            'global_best_position': self.global_best_position,
            'global_best_fitness': self.global_best_fitness,
            'blacklist': self.blacklist,
        }
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f'Không thể lưu file memory: {e}')

    def trigger_gazebo_reset(self):
        """Gọi Gazebo reset."""
        req = Empty.Request()
        if self.reset_world_cli.service_is_ready():
            self.reset_world_cli.call_async(req)
        elif self.reset_sim_cli.service_is_ready():
            self.reset_sim_cli.call_async(req)

    def publish_cmd_vel(self, linear_x):
        """Xuất lệnh điều khiển."""
        twist = Twist()
        twist.linear.x = max(-self.max_velocity, min(self.max_velocity, linear_x))
        self.cmd_vel_pub.publish(twist)

    def _add_to_blacklist(self, reason):
        """Thêm cá thể hiện tại vào blacklist (có giới hạn kích thước)."""
        bad_pos = list(self.particles[self.current_particle_idx].position)
        self.blacklist.append(bad_pos)
        # Tự động cắt bớt nếu quá lớn
        self.blacklist = self._deduplicate_blacklist(self.blacklist)
        self._save_memory()
        self.get_logger().warn(
            f'  ⚠️ Cá thể #{self.current_particle_idx + 1}: {reason} → Blacklist 🚫 '
            f'(Tổng: {len(self.blacklist)}/{self.MAX_BLACKLIST_SIZE} điểm)'
        )

    def _handle_seed_champion_failure(self):
        """
        Khi Hạt #1 (Seed Champion kế thừa từ bộ nhớ) bị ngã, có nghĩa bộ nhớ cũ
        đã bị nhiễm độc. Reset global_best về PID an toàn mặc định để cứu cả bầy đàn.
        """
        if self.current_particle_idx == 0 and self.global_best_fitness < self.MIN_TRUSTWORTHY_FITNESS:
            self.global_best_position = list(self.SAFE_DEFAULT_PID)
            self.global_best_fitness = 0.0
            self.get_logger().warn(
                '  🔄 Seed Champion thất bại! Reset global_best về PID an toàn mặc định [Kp=58, Kd=6.5].'
            )

    def _handle_runaway_fail(self, timestamp):
        """Xử lý loại cá thể khi trôi bạt mạng kịch trần."""
        self.robot_fell = True
        self._add_to_blacklist('TRÔI MẤT KIỂM SOÁT (>= 1.5 m/s)')
        self._handle_seed_champion_failure()
        self._evaluate_and_next_particle(timestamp)

    def imu_callback(self, msg):
        """Xử lý điều khiển và thu thập dữ liệu."""
        pitch = quaternion_to_pitch(msg.orientation)
        gyro_y = msg.angular_velocity.y
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.state_start_time is None:
            self.state_start_time = timestamp

        elapsed = timestamp - self.state_start_time

        # Phát hiện ngã
        if abs(pitch) > self.fall_threshold and self.state not in [self.STATE_RESET, self.STATE_DONE]:
            self.robot_fell = True
            self._add_to_blacklist(f'NGÃ XE (Pitch={math.degrees(pitch):.1f}°)')
            self._handle_seed_champion_failure()
            self._evaluate_and_next_particle(timestamp)
            return

        # ============================================================
        # 1. STATE: RESET (Dựng xe và chờ Gazebo reset ổn định)
        # ============================================================
        if self.state == self.STATE_RESET:
            out = 58.0 * (pitch - 0.0) + 6.5 * gyro_y
            self.publish_cmd_vel(out)

            # Chờ ít nhất 2.0s cho Gazebo reset hoàn tất
            if elapsed < 2.0:
                return

            # Kiểm tra robot ĐÃ THỰC SỰ ĐỨNG chưa (< 10°)
            if abs(pitch) < 0.175:
                # Robot đứng ổn → bắt đầu thử nghiệm
                self._start_particle_trial(timestamp)
            elif elapsed > 5.0:
                # Quá 5 giây mà robot vẫn chưa đứng → gọi lại Gazebo reset
                self.get_logger().warn('  🔄 Robot chưa đứng dậy sau 5s! Gọi lại Gazebo Reset...')
                self.trigger_gazebo_reset()
                self.state_start_time = timestamp  # Reset đồng hồ

        # ============================================================
        # 2. STATE: STATIC (1. Kiểm định độ ổn định tĩnh & rung giật 2.5s)
        # ============================================================
        elif self.state == self.STATE_STATIC:
            error = pitch - self.current_target_pitch
            output = self.current_pid.compute(error, timestamp, measured_rate=gyro_y)
            self.publish_cmd_vel(output)

            self.pitch_history.append(pitch)
            self.static_pitch_history.append(pitch)
            self.output_history.append(output)
            self.chattering_diffs.append(abs(output - self.prev_output))
            self.prev_output = output

            if elapsed > 0.5 and abs(output) >= 1.5:
                self._handle_runaway_fail(timestamp)
                return

            if elapsed > self.balance_duration:
                self.state = self.STATE_DISTURB_FWD
                self.state_start_time = timestamp
                self.get_logger().info('    👉 [Huých 1/2 - TIẾN]: Tác dụng xung lực xô về phía trước (+0.18 m/s)...')

        # ============================================================
        # 3. STATE: DISTURB_FWD (2. Xung lực huých TIẾN 0.10s)
        # ============================================================
        elif self.state == self.STATE_DISTURB_FWD:
            error = pitch - self.current_target_pitch
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

        # ============================================================
        # 4. STATE: RECOVER_FWD (3. Đo phản xạ hồi phục TIẾN: Ts, Mp, ITAE 2.8s)
        # ============================================================
        elif self.state == self.STATE_RECOVER_FWD:
            error = pitch - self.current_target_pitch
            output = self.current_pid.compute(error, timestamp, measured_rate=gyro_y)
            self.publish_cmd_vel(output)

            self.pitch_history.append(pitch)
            self.output_history.append(output)
            self.chattering_diffs.append(abs(output - self.prev_output))
            self.prev_output = output

            self.overshoot_fwd = max(self.overshoot_fwd, abs(pitch))
            self.itae_fwd += elapsed * abs(pitch) * 0.01

            # Settling Time Ts: thời gian kéo góc về an toàn (< 1.5 độ)
            if abs(pitch) < 0.026 and self.settling_time_fwd >= self.recovery_duration and elapsed > 0.2:
                self.settling_time_fwd = elapsed

            if elapsed > 0.8 and abs(output) >= 1.5:
                self._handle_runaway_fail(timestamp)
                return

            if elapsed > self.recovery_duration:
                self.state = self.STATE_DISTURB_BWD
                self.state_start_time = timestamp
                self.get_logger().info('    👉 [Huých 2/2 - LÙI]: Tác dụng xung lực xô về phía sau (-0.18 m/s)...')

        # ============================================================
        # 5. STATE: DISTURB_BWD (4. Xung lực huých LÙI 0.10s - Đối xứng 2 chiều)
        # ============================================================
        elif self.state == self.STATE_DISTURB_BWD:
            error = pitch - self.current_target_pitch
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

        # ============================================================
        # 6. STATE: RECOVER_BWD (5. Đo phản xạ hồi phục LÙI: Ts, Mp, ITAE 2.8s)
        # ============================================================
        elif self.state == self.STATE_RECOVER_BWD:
            error = pitch - self.current_target_pitch
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

            if elapsed > 0.8 and abs(output) >= 1.5:
                self._handle_runaway_fail(timestamp)
                return

            if elapsed > self.recovery_duration:
                self._evaluate_and_next_particle(timestamp)

    def _start_particle_trial(self, timestamp):
        """Bắt đầu thử nghiệm cá thể theo Tiêu chuẩn Kiểm định Điều khiển học."""
        particle = self.particles[self.current_particle_idx]
        kp, ki, kd, target_p = particle.position

        self.current_pid = PIDController(
            kp=kp, ki=ki, kd=kd,
            output_min=-self.max_velocity,
            output_max=self.max_velocity,
            integral_max=5.0,
            derivative_filter_alpha=0.1
        )
        self.current_target_pitch = target_p
        self.pitch_history = []
        self.output_history = []
        self.static_pitch_history = []
        self.chattering_diffs = []
        self.prev_output = 0.0
        self.robot_fell = False

        # Các chỉ số tiêu chuẩn kiểm định IEEE
        self.overshoot_fwd = 0.0
        self.overshoot_bwd = 0.0
        self.settling_time_fwd = self.recovery_duration
        self.settling_time_bwd = self.recovery_duration
        self.itae_fwd = 0.0
        self.itae_bwd = 0.0

        self.state = self.STATE_STATIC
        self.state_start_time = timestamp

        if self.current_particle_idx == 0:
            seed_desc = '👑 SEED VÔ ĐỊCH (Kế thừa từ ổ cứng)'
        else:
            seed_desc = f'🔍 Hạt thăm dò né Blacklist #{self.current_particle_idx}'

        self.get_logger().info(
            f'  [Thế hệ {self.current_generation}/{self.max_generations}] '
            f'Cá thể #{self.current_particle_idx + 1} [{seed_desc}]:\n'
            f'     Kp={kp:.2f} | Ki={ki:.3f} | Kd={kd:.2f} | Target={target_p:+.5f} rad ({math.degrees(target_p):.3f}°)'
        )

    def _evaluate_and_next_particle(self, timestamp):
        """Tổng hợp và chấm điểm theo Tiêu chuẩn Kiểm định Điều khiển học Quốc tế."""
        particle = self.particles[self.current_particle_idx]
        final_pitch_deg = math.degrees(abs(self.pitch_history[-1])) if self.pitch_history else 45.0

        # Nếu đã ngã HOẶC khi hết giờ mà xe vẫn chưa hồi phục (vẫn nghiêng > 4.5° đang trên đà ngã) -> Cho 0 điểm!
        if self.robot_fell or len(self.pitch_history) < 10 or final_pitch_deg > 4.5:
            fitness = 0.0
            if final_pitch_deg > 4.5 and not self.robot_fell:
                self._add_to_blacklist(f'KHÔNG HỒI PHỤC (Góc cuối={final_pitch_deg:.1f}° > 4.5°)')
            rms_deg = 45.0
            mp_deg = 45.0
            ts_sec = self.recovery_duration
            ess_deg = 45.0
            itae = 100.0
            chatter = 1.0
            avg_drift = 1.5
        else:
            # 1. Sai số xác lập tĩnh (Steady-state error e_ss)
            n_stat = len(self.static_pitch_history)
            recent_stat = self.static_pitch_history[-min(50, n_stat):]
            ess_rad = abs(sum(recent_stat) / max(1, len(recent_stat)))
            ess_deg = math.degrees(ess_rad)

            # 2. Độ vọt lố tối đa 2 chiều (Maximum Overshoot Mp)
            mp_rad = max(self.overshoot_fwd, self.overshoot_bwd)
            mp_deg = math.degrees(mp_rad)

            # 3. Thời gian ổn định (Settling Time Ts)
            ts_sec = max(self.settling_time_fwd, self.settling_time_bwd)

            # 4. Chỉ số tích phân sai số chuẩn quốc tế (ITAE)
            itae = self.itae_fwd + self.itae_bwd

            # 5. Độ êm mô-tơ (Actuator Health / Chattering Index)
            chatter = sum(self.chattering_diffs) / max(1, len(self.chattering_diffs))

            # 6. Tốc độ trôi trung bình (Drift)
            avg_drift = abs(sum(self.output_history) / max(1, len(self.output_history)))

            # 7. Độ rung lắc góc toàn bài (RMS Pitch)
            rms_rad = math.sqrt(sum(p**2 for p in self.pitch_history) / len(self.pitch_history))
            rms_deg = math.degrees(rms_rad)

            # Phát hiện xe trôi bạt mạng liên tục (tốc độ trung bình >= 1.4 m/s kịch trần) -> Loại!
            if avg_drift >= 1.4:
                fitness = 0.0
                self._add_to_blacklist(f'TRÔI LIÊN TỤC (Drift TB={avg_drift:.2f} m/s >= 1.4)')
            else:
                # CÔNG THỨC FITNESS TỔNG HỢP CHUẨN ĐIỀU KHIỂN HỌC (THANG 100 ĐIỂM)
                # Đánh giá toàn diện: RMS, Vọt lố Mp, Thời gian Ts, ITAE, Trôi Drift, Rung Chattering
                penalty = (
                    (0.04 * rms_deg) +
                    (0.02 * mp_deg) +
                    (0.12 * ts_sec) +
                    (0.03 * itae) +
                    (0.35 * avg_drift) +
                    (0.40 * chatter)
                )
                fitness = 100.0 * math.exp(-penalty)

        particle.current_fitness = fitness

        # Cập nhật Best cá nhân
        if fitness > particle.best_fitness:
            particle.best_fitness = fitness
            particle.best_position = list(particle.position)

        # Cập nhật Kỷ lục Toàn bầy (Global Best)
        if fitness > self.global_best_fitness:
            self.global_best_fitness = fitness
            self.global_best_position = list(particle.position)
            star = ' ⭐ (KỶ LỤC MỚI!)'
            self._save_memory()
        else:
            star = ''

        if fitness > 0:
            self.get_logger().info(
                f'    📊 [CHUẨN IEEE]: e_ss={ess_deg:.3f}° | Mp={mp_deg:.1f}° | Ts={ts_sec:.2f}s | '
                f'ITAE={itae:.2f} | Chatter={chatter:.3f} | Drift={avg_drift:.2f}m/s'
            )
            self.get_logger().info(f'    ➜ ĐIỂM TIÊU CHUẨN ĐIỀU KHIỂN: {fitness:.1f} / 100{star}')
        else:
            self.get_logger().info(f'    ➜ Điểm: 0.0 / 100 (Không đạt tiêu chuẩn kiểm định)')

        # Chuyển cá thể tiếp theo
        self.current_particle_idx += 1

        if self.current_particle_idx >= self.num_particles:
            self.get_logger().info('')
            self.get_logger().info(
                f'  🏁 KẾT THÚC THẾ HỆ #{self.current_generation}! '
                f'Kỷ lục bầy đàn hiện tại: {self.global_best_fitness:.1f}/100'
            )

            if self.current_generation < self.max_generations:
                self.current_generation += 1
                self.current_particle_idx = 0
                # Cập nhật vị trí cả bầy đàn, tránh vùng cấm blacklist
                for p in self.particles:
                    p.update(self.global_best_position, self.blacklist)
                self.get_logger().info(f'  🚀 Bầy đàn đang hội tụ sang Thế hệ #{self.current_generation}...')
                self.get_logger().info('')
            else:
                self._finish_pso_tuning()
                return

        # Reset Gazebo cho lượt tiếp
        self.state = self.STATE_RESET
        self.state_start_time = timestamp
        self.trigger_gazebo_reset()

    def _finish_pso_tuning(self):
        """Hoàn tất quá trình tối ưu và xuất file YAML."""
        self.state = self.STATE_DONE
        self.publish_cmd_vel(0.0)

        kp, ki, kd, target_p = self.global_best_position
        self._save_memory()

        self.get_logger().info('')
        self.get_logger().info('=' * 70)
        self.get_logger().info('     🏆 KẾT QUẢ TỐI ƯU HÓA BẦY ĐÀN (PSO) HOÀN TẤT 🏆')
        self.get_logger().info('=' * 70)
        self.get_logger().info(f'  Điểm Fitness tối ưu:   {self.global_best_fitness:.1f} / 100')
        self.get_logger().info('')
        self.get_logger().info(f'  ┌──────────────────────────────────────────────────┐')
        self.get_logger().info(f'  │  Kp           = {kp:>10.4f}                       │')
        self.get_logger().info(f'  │  Ki           = {ki:>10.4f}                       │')
        self.get_logger().info(f'  │  Kd           = {kd:>10.4f}                       │')
        self.get_logger().info(f'  │  Target Pitch = {target_p:>+10.5f} rad ({math.degrees(target_p):.3f}°)         │')
        self.get_logger().info(f'  └──────────────────────────────────────────────────┘')
        self.get_logger().info('')
        self.get_logger().info('  ✓ Bộ nhớ học tập đã được lưu vào ~/.pso_pid_memory.json')
        self.get_logger().info('  ✓ File cấu hình tối ưu đã lưu vào ~/tuned_pid_params.yaml')
        self.get_logger().info('=' * 70)

        self._save_yaml(kp, ki, kd, target_p)

    def _save_yaml(self, kp, ki, kd, target_p):
        """Lưu ra file YAML."""
        content = f"""# =============================================
# PID Tuning Results - PSO với Trí nhớ Học liên tục
# Điểm Fitness tối ưu: {self.global_best_fitness:.1f}/100
# =============================================

balance_controller:
  ros__parameters:
    kp: {kp:.4f}
    ki: {ki:.4f}
    kd: {kd:.4f}
    target_pitch: {target_p:.5f}
    max_velocity: 1.5
    integral_max: 5.0
    derivative_filter_alpha: 0.1
    use_gyro_derivative: true
    fall_threshold: 0.785
    enabled: true

    # --- Vòng điều khiển Vận tốc & Vị trí (Chống trôi xe - Cascaded Loop) ---
    enable_velocity_control: true
    kp_velocity: 0.08
    kp_position: 0.015
    max_pitch_adjustment: 0.06
"""
        try:
            with open(self.output_file, 'w') as f:
                f.write(content)
            self.get_logger().info(f'  📁 Đã lưu file cấu hình: {self.output_file}')
        except Exception as e:
            self.get_logger().warn(f'Không thể lưu file yaml: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = PsoPIDTunerNode()
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
