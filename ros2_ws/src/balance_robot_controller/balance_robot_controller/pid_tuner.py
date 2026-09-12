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

        # Cơ chế Né tránh Vùng Xấu (Tabu Avoidance)
        if blacklist:
            for bad_pos in blacklist:
                # Tính khoảng cách chuẩn hóa tới điểm xấu
                dist_sq = sum(
                    ((self.position[k] - bad_pos[k]) / (self.bounds[k][1] - self.bounds[k][0])) ** 2
                    for k in range(len(self.position))
                )
                if dist_sq < 0.04:  # Bán kính nguy hiểm (r < 0.2)
                    # Lực đẩy đẩy hạt bay về phía an toàn (hướng về global_best)
                    for k in range(len(self.position)):
                        self.position[k] = 0.7 * self.position[k] + 0.3 * global_best_pos[k]


class PsoPIDTunerNode(Node):
    """
    ROS 2 Node thực thi thuật toán PSO có Trí nhớ vĩnh viễn (Memory Persistence).
    """

    STATE_RESET = 'RESET'
    STATE_BALANCE = 'BALANCE'         # Kiểm tra đứng yên
    STATE_DISTURBANCE = 'DISTURB'     # Tác dụng xung lực
    STATE_RECOVERY = 'RECOVERY'       # Đo phản hồi hồi phục
    STATE_DONE = 'DONE'

    # Không gian tìm kiếm 4 chiều: [Kp, Ki, Kd, Target_Pitch]
    SEARCH_BOUNDS = [
        (45.0, 75.0),       # Kp
        (0.2, 1.2),         # Ki (nhỏ để chống trôi)
        (4.5, 9.0),         # Kd
        (-0.008, 0.008),    # Target Pitch (rad)
    ]

    def __init__(self):
        super().__init__('pid_tuner')

        # ===== Parameters =====
        self.declare_parameter('num_particles', 4)
        self.declare_parameter('max_generations', 3)
        self.declare_parameter('fall_threshold', 0.785)
        self.declare_parameter('max_velocity', 1.5)
        self.declare_parameter('memory_file', '~/.pso_pid_memory.json')
        self.declare_parameter('output_file', '~/tuned_pid_params.yaml')

        self.num_particles = self.get_parameter('num_particles').value
        self.max_generations = self.get_parameter('max_generations').value
        self.fall_threshold = self.get_parameter('fall_threshold').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.memory_file = os.path.expanduser(self.get_parameter('memory_file').value)
        self.output_file = os.path.expanduser(self.get_parameter('output_file').value)

        # Khởi tạo hoặc nạp bộ nhớ AI từ file
        self.blacklist = []
        self.history_records = []
        self.global_best_position = [58.0, 0.75, 6.5, 0.0]
        self.global_best_fitness = 0.0

        self._load_memory()

        # Khởi tạo bầy đàn
        self.particles = []
        # Hạt #1 luôn là Hạt giống Kỷ lục Tốt nhất (Seed Champion)
        self.particles.append(Particle(self.SEARCH_BOUNDS, initial_pos=self.global_best_position))

        # Các hạt còn lại phân bố tìm kiếm xung quanh
        for _ in range(1, self.num_particles):
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
        """Nạp dữ liệu học từ file nếu đã từng chạy trước đây."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    if 'global_best_position' in data:
                        self.global_best_position = data['global_best_position']
                        self.global_best_fitness = data.get('global_best_fitness', 0.0)
                    self.blacklist = data.get('blacklist', [])
            except Exception as e:
                self.get_logger().warn(f'Không thể đọc file memory: {e}')

    def _save_memory(self):
        """Lưu lại trí nhớ học được vào ổ đĩa."""
        data = {
            'global_best_position': self.global_best_position,
            'global_best_fitness': self.global_best_fitness,
            'blacklist': self.blacklist[-20:],  # Lưu 20 điểm xấu gần nhất
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
            # Thêm điểm làm ngã vào Blacklist để lần sau không bao giờ thử lại
            bad_pos = list(self.particles[self.current_particle_idx].position)
            self.blacklist.append(bad_pos)
            self.get_logger().warn(
                f'  ⚠️ Cá thể #{self.current_particle_idx + 1} làm ngã xe (Pitch = {math.degrees(pitch):.1f}°)! '
                f'Đã đưa vào Blacklist 🚫'
            )
            self._evaluate_and_next_particle(timestamp)
            return

        # ============================================================
        # 1. STATE: RESET (Dựng xe và giữ vững ngay lập tức)
        # ============================================================
        if self.state == self.STATE_RESET:
            out = 58.0 * (pitch - 0.0) + 6.5 * gyro_y
            self.publish_cmd_vel(out)
            if elapsed > 0.8:
                self._start_particle_trial(timestamp)

        # ============================================================
        # 2. STATE: BALANCE (Kiểm tra đứng yên 2.0s)
        # ============================================================
        elif self.state == self.STATE_BALANCE:
            error = pitch - self.current_target_pitch
            output = self.current_pid.compute(error, timestamp, measured_rate=gyro_y)
            self.publish_cmd_vel(output)

            self.pitch_history.append(pitch)
            self.output_history.append(output)

            if elapsed > 2.0:
                self.state = self.STATE_DISTURBANCE
                self.state_start_time = timestamp
                self.get_logger().info('    👉 [Lực đẩy]: Tác dụng lực xô thử nghiệm vào xe...')

        # ============================================================
        # 3. STATE: DISTURBANCE (Xung lực 0.12s)
        # ============================================================
        elif self.state == self.STATE_DISTURBANCE:
            self.publish_cmd_vel(0.30)
            if elapsed > 0.12:
                self.state = self.STATE_RECOVERY
                self.state_start_time = timestamp
                self.max_recovery_pitch = 0.0

        # ============================================================
        # 4. STATE: RECOVERY (Đo phản xạ kéo lại thăng bằng 2.5s)
        # ============================================================
        elif self.state == self.STATE_RECOVERY:
            error = pitch - self.current_target_pitch
            output = self.current_pid.compute(error, timestamp, measured_rate=gyro_y)
            self.publish_cmd_vel(output)

            self.pitch_history.append(pitch)
            self.output_history.append(output)
            self.max_recovery_pitch = max(self.max_recovery_pitch, abs(pitch))

            if elapsed > 2.5:
                self._evaluate_and_next_particle(timestamp)

    def _start_particle_trial(self, timestamp):
        """Bắt đầu thử nghiệm cá thể hiện tại."""
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
        self.robot_fell = False
        self.max_recovery_pitch = 0.0

        self.state = self.STATE_BALANCE
        self.state_start_time = timestamp

        self.get_logger().info(
            f'  [Thế hệ {self.current_generation}/{self.max_generations}] '
            f'Cá thể #{self.current_particle_idx + 1}: '
            f'Kp={kp:.2f} | Ki={ki:.3f} | Kd={kd:.2f} | Target={target_p:+.4f}'
        )

    def _evaluate_and_next_particle(self, timestamp):
        """Chấm điểm theo thang điểm 100% chuẩn xác và tiến hóa bầy đàn."""
        particle = self.particles[self.current_particle_idx]

        if self.robot_fell or len(self.pitch_history) < 10:
            fitness = 0.0
            rms_deg = 45.0
            over_deg = 45.0
        else:
            rms_pitch = math.sqrt(sum(p**2 for p in self.pitch_history) / len(self.pitch_history))
            rms_deg = math.degrees(rms_pitch)
            over_deg = math.degrees(self.max_recovery_pitch)
            avg_drift = abs(sum(self.output_history) / len(self.output_history))

            # THANG ĐIỂM HÀM MŨ CHUẨN 0 - 100%
            # Đứng vững (RMS < 1°), kháng lực tốt (vọt lố < 3°), không trôi => Điểm 85 ~ 98/100
            penalty = (0.12 * rms_deg) + (0.04 * over_deg) + (1.2 * avg_drift)
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
            # Lưu ngay vào file nhớ
            self._save_memory()
        else:
            star = ''

        self.get_logger().info(f'    ➜ Độ vững: RMS={rms_deg:.2f}° | Điểm: {fitness:.1f}/100{star}')

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
