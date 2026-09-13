"""
Balance Controller Node - Bộ điều khiển cân bằng xe hai bánh.

Sử dụng PID controller để giữ robot đứng thẳng bằng cách đọc dữ liệu
cảm biến IMU và điều khiển vận tốc bánh xe.

Subscribes:
    /imu/data (sensor_msgs/Imu) - Dữ liệu cảm biến IMU (MPU6050)

Publishes:
    /cmd_vel (geometry_msgs/Twist) - Lệnh vận tốc cho differential drive
    /balance/diagnostics (std_msgs/Float64MultiArray) - Dữ liệu chẩn đoán PID
    /balance/status (std_msgs/String) - Trạng thái controller

Parameters (có thể thay đổi runtime bằng ros2 param set):
    kp, ki, kd          : Hệ số PID
    target_pitch         : Góc pitch mục tiêu (rad, 0 = đứng thẳng)
    max_velocity         : Vận tốc tối đa (m/s)
    integral_max         : Giới hạn anti-windup
    use_gyro_derivative  : Dùng gyro cho đạo hàm (khuyến nghị: true)
    fall_threshold       : Ngưỡng phát hiện ngã (rad)
    enabled              : Bật/tắt controller
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, Float64MultiArray, MultiArrayDimension, String

from balance_robot_controller.pid import PIDController


def quaternion_to_pitch(q):
    """
    Trích xuất góc pitch từ quaternion (IMU orientation).
    Pitch = góc xoay quanh trục Y (nghiêng trước/sau).

    Args:
        q: Quaternion (geometry_msgs/Quaternion)

    Returns:
        Góc pitch (radians)
    """
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    if abs(sinp) >= 1.0:
        return math.copysign(math.pi / 2.0, sinp)
    return math.asin(sinp)


class BalanceControllerNode(Node):
    """
    ROS2 Node điều khiển cân bằng xe hai bánh.

    Nguyên lý hoạt động:
    1. Đọc góc pitch từ IMU (quaternion → euler)
    2. Tính sai số: error = target_pitch - current_pitch
    3. PID controller tính vận tốc cần thiết
    4. Publish vận tốc lên /cmd_vel → diff_drive plugin → bánh xe
    """

    def __init__(self):
        super().__init__('balance_controller')

        # ===== Khai báo parameters =====
        self.declare_parameter('kp', 48.0)
        self.declare_parameter('ki', 0.3)
        self.declare_parameter('kd', 7.8)
        self.declare_parameter('target_pitch', 0.0)
        self.declare_parameter('max_velocity', 2)
        self.declare_parameter('integral_max', 10.0)
        self.declare_parameter('derivative_filter_alpha', 1)
        self.declare_parameter('use_gyro_derivative', True)
        self.declare_parameter('fall_threshold', 0.785)  # ~45 degrees
        self.declare_parameter('enabled', True)

        # --- Vòng điều khiển Vận tốc & Vị trí (Cascaded Loop - Chống trôi xe) ---
        self.declare_parameter('enable_velocity_control', True)
        self.declare_parameter('kp_velocity', 0.08)           # Giảm chấn vận tốc êm dịu, không giật
        self.declare_parameter('kp_position', 0.015)          # Lực kéo vị trí nhẹ nhàng, chống lắc lư
        self.declare_parameter('max_pitch_adjustment', 0.06)  # ~3.5 độ (giới hạn góc ngửa êm ái)

        # ===== Lấy giá trị ban đầu =====
        kp = self.get_parameter('kp').value
        ki = self.get_parameter('ki').value
        kd = self.get_parameter('kd').value
        max_vel = self.get_parameter('max_velocity').value
        integral_max = self.get_parameter('integral_max').value
        alpha = self.get_parameter('derivative_filter_alpha').value

        # ===== Tạo PID controller =====
        self.pid = PIDController(
            kp=kp, ki=ki, kd=kd,
            output_min=-max_vel, output_max=max_vel,
            integral_max=integral_max,
            derivative_filter_alpha=alpha
        )

        self.target_pitch = self.get_parameter('target_pitch').value
        self.use_gyro = self.get_parameter('use_gyro_derivative').value
        self.fall_threshold = self.get_parameter('fall_threshold').value
        self.enabled = self.get_parameter('enabled').value
        self.fallen = False
        self.msg_count = 0

        # Tham số & biến trạng thái vòng chống trôi
        self.enable_velocity_control = self.get_parameter('enable_velocity_control').value
        self.kp_velocity = self.get_parameter('kp_velocity').value
        self.kp_position = self.get_parameter('kp_position').value
        self.max_pitch_adjustment = self.get_parameter('max_pitch_adjustment').value
        self.current_vel_x = 0.0
        self.current_pos_x = 0.0
        self.target_pos_x = None

        # ===== Callback cập nhật parameter runtime =====
        self.add_on_set_parameters_callback(self.parameter_callback)

        # ===== Subscribers (Dùng qos_profile_sensor_data chống rớt gói) =====
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, qos_profile_sensor_data
        )
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, qos_profile_sensor_data
        )

        # ===== Publishers =====
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.target_pitch_pub = self.create_publisher(Float64, 'balance/target_pitch', 10)
        self.diag_pub = self.create_publisher(
            Float64MultiArray, 'balance/diagnostics', 10
        )
        self.status_pub = self.create_publisher(String, 'balance/status', 10)

        # Các topic Float64 riêng biệt cho rqt_plot
        self.pitch_pub = self.create_publisher(Float64, 'balance/pitch', 10)
        self.error_pub = self.create_publisher(Float64, 'balance/error', 10)
        self.p_pub = self.create_publisher(Float64, 'balance/p_term', 10)
        self.i_pub = self.create_publisher(Float64, 'balance/i_term', 10)
        self.d_pub = self.create_publisher(Float64, 'balance/d_term', 10)
        self.output_pub = self.create_publisher(Float64, 'balance/output', 10)

        # ===== Log thông tin khởi động =====
        self.get_logger().info('=' * 50)
        self.get_logger().info('  BALANCE CONTROLLER - Bộ điều khiển cân bằng')
        self.get_logger().info('=' * 50)
        self.get_logger().info(f'  PID: Kp={kp}, Ki={ki}, Kd={kd}')
        self.get_logger().info(f'  Target pitch: {self.target_pitch} rad')
        self.get_logger().info(f'  Max velocity: {max_vel} m/s')
        self.get_logger().info(f'  Gyro derivative: {self.use_gyro}')
        self.get_logger().info(f'  Fall threshold: {math.degrees(self.fall_threshold):.1f}°')
        self.get_logger().info('=' * 50)
        self.get_logger().info('Đang chờ dữ liệu IMU...')

        self.publish_status('WAITING')

    def parameter_callback(self, params):
        """
        Callback xử lý thay đổi parameter tại runtime.
        Cho phép người dùng tune PID bằng lệnh:
            ros2 param set /balance_controller kp 60.0
            ros2 param set /balance_controller ki 1.0
            ros2 param set /balance_controller kd 8.0
        """
        for param in params:
            name = param.name
            value = param.value

            if name == 'kp':
                self.pid.kp = value
                self.get_logger().info(f'>> Kp = {value}')
            elif name == 'ki':
                self.pid.ki = value
                self.get_logger().info(f'>> Ki = {value}')
            elif name == 'kd':
                self.pid.kd = value
                self.get_logger().info(f'>> Kd = {value}')
            elif name == 'target_pitch':
                self.target_pitch = value
                self.get_logger().info(f'>> Target pitch = {value} rad')
            elif name == 'max_velocity':
                self.pid.output_min = -value
                self.pid.output_max = value
                self.get_logger().info(f'>> Max velocity = {value} m/s')
            elif name == 'integral_max':
                self.pid.integral_max = value
            elif name == 'derivative_filter_alpha':
                self.pid.alpha = value
            elif name == 'use_gyro_derivative':
                self.use_gyro = value
            elif name == 'fall_threshold':
                self.fall_threshold = value
            elif name == 'enable_velocity_control':
                self.enable_velocity_control = value
                self.get_logger().info(f'>> Enable velocity control = {value}')
            elif name == 'kp_velocity':
                self.kp_velocity = value
                self.get_logger().info(f'>> Kp velocity = {value}')
            elif name == 'kp_position':
                self.kp_position = value
                self.get_logger().info(f'>> Kp position = {value}')
            elif name == 'max_pitch_adjustment':
                self.max_pitch_adjustment = value
                self.get_logger().info(f'>> Max pitch adjustment = {value} rad')
            elif name == 'enabled':
                self.enabled = value
                if value:
                    self.pid.reset()
                    self.fallen = False
                    self.target_pos_x = self.current_pos_x
                    self.publish_status('RUNNING')
                    self.get_logger().info('Controller ENABLED')
                else:
                    self.publish_cmd_vel(0.0)
                    self.publish_status('DISABLED')
                    self.get_logger().info('Controller DISABLED')

        return SetParametersResult(successful=True)

    def odom_callback(self, msg):
        """Callback cập nhật vận tốc và vị trí xe từ Odometry có lọc thông thấp làm mượt."""
        raw_vel = msg.twist.twist.linear.x
        # Lọc thông thấp (75% cũ + 25% mới) để triệt tiêu rung giật
        self.current_vel_x = 0.75 * self.current_vel_x + 0.25 * raw_vel
        self.current_pos_x = msg.pose.pose.position.x
        if self.target_pos_x is None:
            self.target_pos_x = self.current_pos_x

    def imu_callback(self, msg):
        """
        Callback xử lý dữ liệu IMU.
        Được gọi mỗi khi nhận message từ /imu/data (100Hz).
        """
        if not self.enabled:
            return

        # Log lần đầu nhận IMU data
        self.msg_count += 1
        if self.msg_count == 1:
            self.get_logger().info('Đã nhận dữ liệu IMU! Controller đang chạy.')
            self.publish_status('RUNNING')

        # Trích xuất góc pitch từ quaternion
        pitch = quaternion_to_pitch(msg.orientation)

        # ===== Phát hiện ngã =====
        if abs(pitch) > self.fall_threshold:
            if not self.fallen:
                self.fallen = True
                self.target_pos_x = None  # Reset lại vị trí mục tiêu khi ngã
                self.get_logger().warn(
                    f'ROBOT ĐÃ NGÃ! Pitch = {math.degrees(pitch):.1f}°. '
                    f'Dừng động cơ. Đặt enabled=false rồi true để reset.'
                )
                self.publish_status('FALLEN')
            self.publish_cmd_vel(0.0)
            self.publish_diagnostics(pitch, pitch - self.target_pitch, 0.0)
            return
        else:
            if self.fallen:
                # Robot được dựng lại
                self.fallen = False
                self.pid.reset()
                self.target_pos_x = self.current_pos_x  # Đặt điểm đứng mới làm gốc
                self.get_logger().info('Robot đã dựng lại. Tiếp tục cân bằng.')
                self.publish_status('RUNNING')

        # ===== Vòng lặp kép Cascaded PID (Chống trôi xe có Deadband triệt tiêu dao động) =====
        if self.enable_velocity_control and self.target_pos_x is not None:
            vel_error = self.current_vel_x
            pos_error = self.current_pos_x - self.target_pos_x

            # Deadband (Vùng chết dung sai nhỏ):
            # Nếu xe chỉ nhích siêu nhẹ (< 1.5 cm hoặc < 0.03 m/s) thì coi như đứng yên,
            # KHÔNG bù góc để triệt tiêu hoàn toàn hiện tượng lắc lư qua lại tại một chỗ!
            if abs(vel_error) < 0.03:
                vel_error = 0.0
            if abs(pos_error) < 0.015:
                pos_error = 0.0

            # Khi xe trôi tới (v > 0, x > 0) -> tăng pitch_adjust dương để error = pitch - target bị giảm âm
            # -> output giảm âm -> bánh xe tự động phanh hãm và lùi lại vị trí gốc!
            pitch_adjust = + (self.kp_velocity * vel_error + self.kp_position * pos_error)
            # Kẹp góc bù an toàn (mặc định tối đa ±3.5 độ)
            pitch_adjust = max(-self.max_pitch_adjustment, min(self.max_pitch_adjustment, pitch_adjust))
        else:
            pitch_adjust = 0.0

        effective_target_pitch = self.target_pitch + pitch_adjust
        self.target_pitch_pub.publish(Float64(data=effective_target_pitch))

        # ===== Tính sai số (pitch > target => xe tiến để đón trọng tâm) =====
        error = pitch - effective_target_pitch

        # ===== Lấy timestamp =====
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # ===== Tính PID output =====
        if self.use_gyro:
            # Dùng gyroscope trục Y làm đạo hàm (chính xác hơn vi phân)
            gyro_y = msg.angular_velocity.y
            output = self.pid.compute(error, timestamp, measured_rate=gyro_y)
        else:
            output = self.pid.compute(error, timestamp)

        # ===== Publish =====
        self.publish_cmd_vel(output)
        self.publish_diagnostics(pitch, error, output)

    def publish_cmd_vel(self, linear_x):
        """Publish lệnh vận tốc lên /cmd_vel."""
        twist = Twist()
        twist.linear.x = linear_x
        self.cmd_vel_pub.publish(twist)

    def publish_diagnostics(self, pitch, error, output):
        """
        Publish dữ liệu chẩn đoán lên /balance/diagnostics.

        Data layout (Float64MultiArray):
            [0] pitch         - Góc pitch hiện tại (rad)
            [1] error         - Sai số pitch (rad)
            [2] p_term        - Thành phần P
            [3] i_term        - Thành phần I
            [4] d_term        - Thành phần D
            [5] output        - Đầu ra PID (m/s)
            [6] integral      - Giá trị tích phân

        Dùng rqt_plot hoặc ros2 topic echo để theo dõi:
            ros2 topic echo /balance/diagnostics
        """
        msg = Float64MultiArray()
        msg.layout.dim = [MultiArrayDimension(
            label='fields', size=7, stride=7
        )]
        msg.data = [
            pitch,
            error,
            self.pid.last_p_term,
            self.pid.last_i_term,
            self.pid.last_d_term,
            output,
            self.pid.integral,
        ]
        self.diag_pub.publish(msg)

        # Publish các topic Float64 đơn lẻ để rqt_plot nhận diện ngay lập tức
        f_pitch = Float64()
        f_pitch.data = float(pitch)
        self.pitch_pub.publish(f_pitch)

        f_error = Float64()
        f_error.data = float(error)
        self.error_pub.publish(f_error)

        f_p = Float64()
        f_p.data = float(self.pid.last_p_term)
        self.p_pub.publish(f_p)

        f_i = Float64()
        f_i.data = float(self.pid.last_i_term)
        self.i_pub.publish(f_i)

        f_d = Float64()
        f_d.data = float(self.pid.last_d_term)
        self.d_pub.publish(f_d)

        f_out = Float64()
        f_out.data = float(output)
        self.output_pub.publish(f_out)

    def publish_status(self, status):
        """Publish trạng thái controller lên /balance/status."""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BalanceControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        # Dừng động cơ trước khi thoát
        twist = Twist()
        node.cmd_vel_pub.publish(twist)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
