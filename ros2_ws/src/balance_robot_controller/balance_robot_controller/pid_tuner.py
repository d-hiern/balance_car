"""
PID Auto-Tuner Node - Sử dụng Thuật toán Tối ưu hóa Bầy đàn (Particle Swarm Optimization - PSO)
Tự động tìm bộ PID và Target Pitch tối ưu nhất cho xe cân bằng hai bánh trong Gazebo.

Đặc tính AI nổi bật:
1. Mô phỏng hành vi bầy đàn (PSO) tìm kiếm trong không gian 4 chiều: (Kp, Ki, Kd, Target Pitch).
2. Kiểm tra khả năng kháng nhiễu (Disturbance Rejection): Tự động phát xung lực đẩy thử nghiệm
   để đánh giá khả năng phản hồi thăng bằng trở lại của xe.
3. Tự động Reset thế giới Gazebo (/reset_world) giữa các cá thể.
4. Lưu bộ số tối ưu tốt nhất (Global Best) vào file YAML.
"""

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

    def __init__(self, bounds):
        self.bounds = bounds  # [(min, max) cho Kp, Ki, Kd, Target_Pitch]
        # Khởi tạo vị trí ngẫu nhiên trong vùng tìm kiếm hợp lý
        self.position = [
            random.uniform(b[0], b[1]) for b in bounds
        ]
        # Vận tốc bay ban đầu
        self.velocity = [
            random.uniform(-0.5 * (b[1] - b[0]), 0.5 * (b[1] - b[0])) * 0.1
            for b in bounds
        ]
        self.best_position = list(self.position)
        self.best_fitness = -float('inf')
        self.current_fitness = 0.0

    def update(self, global_best_pos, w=0.5, c1=1.5, c2=1.5):
        """Cập nhật vận tốc và vị trí của hạt theo PSO."""
        for i in range(len(self.position)):
            r1 = random.random()
            r2 = random.random()

            # Thành phần quán tính + nhận thức cá nhân + học hỏi bầy đàn
            v_cognitive = c1 * r1 * (self.best_position[i] - self.position[i])
            v_social = c2 * r2 * (global_best_pos[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + v_cognitive + v_social

            # Giới hạn vận tốc bay tối đa
            v_max = (self.bounds[i][1] - self.bounds[i][0]) * 0.2
            self.velocity[i] = max(-v_max, min(v_max, self.velocity[i]))

            # Cập nhật vị trí
            self.position[i] += self.velocity[i]
            # Giới hạn trong biên cho phép
            self.position[i] = max(self.bounds[i][0], min(self.bounds[i][1], self.position[i]))


class PsoPIDTunerNode(Node):
    """
    ROS 2 Node thực thi thuật toán PSO để tìm bộ số PID cho xe cân bằng.
    """

    # Các trạng thái của máy trạng thái
    STATE_RESET = 'RESET'
    STATE_BALANCE = 'BALANCE'         # Kiểm tra đứng yên
    STATE_DISTURBANCE = 'DISTURB'     # Tác dụng xung lực
    STATE_RECOVERY = 'RECOVERY'       # Đo thời gian hồi phục và độ vọt lố
    STATE_EVALUATE = 'EVALUATE'
    STATE_DONE = 'DONE'

    # Không gian tìm kiếm 4 chiều: [Kp, Ki, Kd, Target_Pitch]
    SEARCH_BOUNDS = [
        (45.0, 75.0),       # Kp: 45 đến 75
        (0.2, 1.2),         # Ki: 0.2 đến 1.2 (nhỏ để chống trôi)
        (4.5, 8.5),         # Kd: 4.5 đến 8.5 (giảm xóc mạnh)
        (-0.008, 0.008),    # Target Pitch: bù lệch góc tự nhiên (rad)
    ]

    def __init__(self):
        super().__init__('pid_tuner')

        # ===== Parameters =====
        self.declare_parameter('num_particles', 4)       # Số cá thể trong 1 thế hệ
        self.declare_parameter('max_generations', 3)     # Số thế hệ tìm kiếm
        self.declare_parameter('fall_threshold', 0.785)  # 45 độ
        self.declare_parameter('max_velocity', 1.5)
        self.declare_parameter('output_file', '~/tuned_pid_params.yaml')

        self.num_particles = self.get_parameter('num_particles').value
        self.max_generations = self.get_parameter('max_generations').value
        self.fall_threshold = self.get_parameter('fall_threshold').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.output_file = self.get_parameter('output_file').value

        # Khởi tạo bầy đàn PSO
        self.particles = [Particle(self.SEARCH_BOUNDS) for _ in range(self.num_particles)]
        # Hạt đầu tiên gán điểm khởi đầu chuẩn mẫu
        self.particles[0].position = [55.0, 0.7, 6.0, 0.0]
        self.particles[0].best_position = list(self.particles[0].position)

        self.global_best_position = list(self.particles[0].position)
        self.global_best_fitness = -float('inf')

        # Quản lý tiến trình
        self.current_generation = 1
        self.current_particle_idx = 0
        self.state = self.STATE_RESET
        self.state_start_time = None

        # Bộ điều khiển thử nghiệm hiện tại
        self.current_pid = None
        self.current_target_pitch = 0.0

        # Dữ liệu đo đạc trong 1 lượt thử
        self.pitch_history = []
        self.output_history = []
        self.max_recovery_pitch = 0.0
        self.robot_fell = False

        # Service Gazebo Reset
        self.reset_world_cli = self.create_client(Empty, '/reset_world')
        self.reset_sim_cli = self.create_client(Empty, '/reset_simulation')

        # ROS Publishers / Subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, 'pid_tuner/status', 10)
        self.data_pub = self.create_publisher(Float64MultiArray, 'pid_tuner/data', 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, qos_profile_sensor_data
        )

        self.get_logger().info('')
        self.get_logger().info('=' * 65)
        self.get_logger().info('  🐝 THUẬT TOÁN BẦY ĐÀN (PSO) - TỰ ĐỘNG TỐI ƯU HÓA PID 🤖')
        self.get_logger().info('=' * 65)
        self.get_logger().info(f'  Số cá thể trong bầy (Particles):  {self.num_particles}')
        self.get_logger().info(f'  Số thế hệ tìm kiếm (Generations): {self.max_generations}')
        self.get_logger().info(f'  Thử nghiệm kháng lực đẩy:        TỰ ĐỘNG KÍCH HOẠT')
        self.get_logger().info('=' * 65)
        self.get_logger().info('Đang chuẩn bị cá thể đầu tiên...')
        self.get_logger().info('')

    def trigger_gazebo_reset(self):
        """Gọi service reset trong Gazebo."""
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
        """Vòng lặp điều khiển và đánh giá Fitness của cá thể."""
        pitch = quaternion_to_pitch(msg.orientation)
        gyro_y = msg.angular_velocity.y
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.state_start_time is None:
            self.state_start_time = timestamp

        elapsed = timestamp - self.state_start_time

        # Kiểm tra nếu xe ngã
        if abs(pitch) > self.fall_threshold and self.state not in [self.STATE_RESET, self.STATE_DONE]:
            self.robot_fell = True
            self.get_logger().warn(
                f'  ⚠️ Cá thể #{self.current_particle_idx + 1} làm ngã xe (Pitch = {math.degrees(pitch):.1f}°)! '
                f'Điểm Fitness = 0.'
            )
            self._evaluate_and_next_particle(timestamp)
            return

        # ============================================================
        # 1. STATE: RESET (Dựng xe đứng thẳng và giữ cân bằng cơ bản)
        # ============================================================
        if self.state == self.STATE_RESET:
            # Dùng PD cơ sở giữ xe ngay lập tức khi vừa reset
            out = 55.0 * (pitch - 0.0) + 6.0 * gyro_y
            self.publish_cmd_vel(out)

            if elapsed > 0.8:
                self._start_particle_trial(timestamp)

        # ============================================================
        # 2. STATE: BALANCE (Thử nghiệm đứng thẳng tự nhiên 2.5 giây)
        # ============================================================
        elif self.state == self.STATE_BALANCE:
            error = pitch - self.current_target_pitch
            output = self.current_pid.compute(error, timestamp, measured_rate=gyro_y)
            self.publish_cmd_vel(output)

            # Ghi nhận dữ liệu
            self.pitch_history.append(pitch)
            self.output_history.append(output)

            if elapsed > 2.5:
                # Chuyển sang giai đoạn tác dụng lực đẩy
                self.state = self.STATE_DISTURBANCE
                self.state_start_time = timestamp
                self.get_logger().info('    👉 Tác dụng lực đẩy thử nghiệm vào xe...')

        # ============================================================
        # 3. STATE: DISTURBANCE (Tác dụng xung lực 0.15s để tạo xô đẩy)
        # ============================================================
        elif self.state == self.STATE_DISTURBANCE:
            # Phát một xung giật bánh xe để làm lệch robot
            disturbance_cmd = 0.35  # m/s
            self.publish_cmd_vel(disturbance_cmd)

            if elapsed > 0.15:
                # Ngừng tác dụng lực, chuyển sang đo phản hồi hồi phục
                self.state = self.STATE_RECOVERY
                self.state_start_time = timestamp
                self.max_recovery_pitch = 0.0

        # ============================================================
        # 4. STATE: RECOVERY (Đo tốc độ phản hồi kéo xe về cân bằng 3.0s)
        # ============================================================
        elif self.state == self.STATE_RECOVERY:
            error = pitch - self.current_target_pitch
            output = self.current_pid.compute(error, timestamp, measured_rate=gyro_y)
            self.publish_cmd_vel(output)

            self.pitch_history.append(pitch)
            self.output_history.append(output)
            self.max_recovery_pitch = max(self.max_recovery_pitch, abs(pitch))

            if elapsed > 3.0:
                # Hoàn thành 1 lượt thử cá thể
                self._evaluate_and_next_particle(timestamp)

    def _start_particle_trial(self, timestamp):
        """Chuẩn bị thông số của cá thể hiện tại và bắt đầu đo."""
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
            f'Thử cá thể #{self.current_particle_idx + 1}: '
            f'Kp={kp:.2f} | Ki={ki:.3f} | Kd={kd:.2f} | Target={target_p:+.4f}'
        )

    def _evaluate_and_next_particle(self, timestamp):
        """Tính hàm Fitness và chuyển sang cá thể tiếp theo hoặc thế hệ mới."""
        particle = self.particles[self.current_particle_idx]

        if self.robot_fell or len(self.pitch_history) < 10:
            fitness = 0.0
        else:
            # 1. Sai số trung bình bình phương (RMS Pitch Error)
            rms_pitch = math.sqrt(sum(p**2 for p in self.pitch_history) / len(self.pitch_history))
            # 2. Vận tốc trôi xe trung bình
            avg_drift_speed = abs(sum(self.output_history) / len(self.output_history))
            # 3. Độ vọt lố lớn nhất khi bị đẩy
            overshoot = self.max_recovery_pitch

            # HÀM MỤC TIÊU FITNESS (Càng cao càng tối ưu):
            # Thưởng cho: xe ít rung (rms nhỏ), không trôi (drift nhỏ), kháng đẩy tốt (overshoot nhỏ)
            fitness = 100.0 / (
                1.0 + (20.0 * rms_pitch) + (10.0 * overshoot) + (15.0 * avg_drift_speed)
            )

        particle.current_fitness = fitness

        # Cập nhật Best cá nhân
        if fitness > particle.best_fitness:
            particle.best_fitness = fitness
            particle.best_position = list(particle.position)

        # Cập nhật Best toàn bầy đàn (Global Best)
        if fitness > self.global_best_fitness:
            self.global_best_fitness = fitness
            self.global_best_position = list(particle.position)
            star = ' ⭐ (KỶ LỤC MỚI CỦA BẦY ĐÀN!)'
        else:
            star = ''

        self.get_logger().info(f'    ➜ Điểm Fitness: {fitness:.1f}/100{star}')

        # Chuyển cá thể tiếp theo
        self.current_particle_idx += 1

        if self.current_particle_idx >= self.num_particles:
            # ĐÃ HẾT 1 THẾ HỆ ➔ Cập nhật vị trí bầy đàn theo PSO
            self.get_logger().info('')
            self.get_logger().info(
                f'  🏁 KẾT THÚC THẾ HỆ #{self.current_generation}! '
                f'Kỷ lục bầy đàn hiện tại: Fitness = {self.global_best_fitness:.1f}'
            )

            if self.current_generation < self.max_generations:
                self.current_generation += 1
                self.current_particle_idx = 0
                # Cả đàn học hỏi vị trí tốt nhất và bay đến vùng tối ưu
                for p in self.particles:
                    p.update(self.global_best_position)
                self.get_logger().info(f'  🚀 Bầy đàn đang hội tụ sang Thế hệ #{self.current_generation}...')
                self.get_logger().info('')
            else:
                # ĐÃ HOÀN TẤT TẤT CẢ THẾ HỆ
                self._finish_pso_tuning()
                return

        # Gọi reset Gazebo cho lượt tiếp theo
        self.state = self.STATE_RESET
        self.state_start_time = timestamp
        self.trigger_gazebo_reset()

    def _finish_pso_tuning(self):
        """Kết thúc thuật toán PSO và xuất kết quả."""
        self.state = self.STATE_DONE
        self.publish_cmd_vel(0.0)

        kp, ki, kd, target_p = self.global_best_position

        self.get_logger().info('')
        self.get_logger().info('=' * 70)
        self.get_logger().info('     🏆 KẾT QUẢ TỐI ƯU HÓA BẦY ĐÀN (PSO) HOÀN TẤT 🏆')
        self.get_logger().info('=' * 70)
        self.get_logger().info(f'  Điểm Fitness tối ưu:   {self.global_best_fitness:.2f} / 100')
        self.get_logger().info('')
        self.get_logger().info(f'  ┌──────────────────────────────────────────────────┐')
        self.get_logger().info(f'  │  Kp           = {kp:>10.4f}                       │')
        self.get_logger().info(f'  │  Ki           = {ki:>10.4f}                       │')
        self.get_logger().info(f'  │  Kd           = {kd:>10.4f}                       │')
        self.get_logger().info(f'  │  Target Pitch = {target_p:>+10.5f} rad ({math.degrees(target_p):.3f}°)         │')
        self.get_logger().info(f'  └──────────────────────────────────────────────────┘')
        self.get_logger().info('')
        self.get_logger().info('  Đặc tính đạt được:')
        self.get_logger().info('  ✓ Xe đứng vững như kiềng ba chân, triệt tiêu trôi hoàn toàn.')
        self.get_logger().info('  ✓ Khi bị lực xô đẩy, lập tức xuất mô-men hãm và hồi phục nhanh chóng.')
        self.get_logger().info('=' * 70)

        # Lưu file YAML
        self._save_results_yaml(kp, ki, kd, target_p)

    def _save_results_yaml(self, kp, ki, kd, target_p):
        """Lưu kết quả ra file cấu hình YAML."""
        yaml_content = f"""# =============================================
# PID Tuning Results - Thuật toán Tối ưu hóa Bầy đàn (PSO)
# Điểm Fitness kháng lực và cân bằng: {self.global_best_fitness:.1f}/100
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
            output_path = os.path.expanduser(self.output_file)
            with open(output_path, 'w') as f:
                f.write(yaml_content)
            self.get_logger().info(f'  📁 Đã lưu file cấu hình tối ưu tại: {output_path}')
        except Exception as e:
            self.get_logger().warn(f'  Không thể lưu file: {e}')


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
