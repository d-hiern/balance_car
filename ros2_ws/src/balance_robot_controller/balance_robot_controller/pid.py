"""
PID Controller Module
Bộ điều khiển PID với các tính năng:
- Anti-windup (chống bão hòa tích phân)
- Derivative filtering (lọc nhiễu đạo hàm)
- Output clamping (giới hạn đầu ra)
- Hỗ trợ gyroscope-based derivative
"""

import math


class PIDController:
    """
    Bộ điều khiển PID hoàn chỉnh.

    Công thức:
        output = Kp * error + Ki * ∫error·dt + Kd * d(error)/dt

    Tính năng:
        - Anti-windup: giới hạn giá trị tích phân
        - Low-pass filter trên đạo hàm để giảm nhiễu
        - Hỗ trợ gyro: dùng tốc độ góc đo trực tiếp thay vì vi phân error
    """

    def __init__(self, kp=0.0, ki=0.0, kd=0.0,
                 output_min=-float('inf'), output_max=float('inf'),
                 integral_max=float('inf'),
                 derivative_filter_alpha=0.1):
        """
        Khởi tạo PID Controller.

        Args:
            kp: Hệ số tỉ lệ (Proportional gain)
            ki: Hệ số tích phân (Integral gain)
            kd: Hệ số đạo hàm (Derivative gain)
            output_min: Giá trị đầu ra tối thiểu
            output_max: Giá trị đầu ra tối đa
            integral_max: Giới hạn tuyệt đối tích phân (anti-windup)
            derivative_filter_alpha: Hệ số lọc thông thấp cho đạo hàm (0-1,
                                     nhỏ hơn = lọc mạnh hơn)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_max = integral_max
        self.alpha = derivative_filter_alpha

        # Trạng thái nội bộ
        self._integral = 0.0
        self._prev_error = None
        self._filtered_derivative = 0.0
        self._prev_time = None

        # Giá trị cuối cùng (cho chẩn đoán)
        self.last_p_term = 0.0
        self.last_i_term = 0.0
        self.last_d_term = 0.0
        self.last_output = 0.0

    def reset(self):
        """Reset toàn bộ trạng thái controller."""
        self._integral = 0.0
        self._prev_error = None
        self._filtered_derivative = 0.0
        self._prev_time = None
        self.last_p_term = 0.0
        self.last_i_term = 0.0
        self.last_d_term = 0.0
        self.last_output = 0.0

    def update_gains(self, kp, ki, kd):
        """Cập nhật hệ số PID tại runtime."""
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def compute(self, error, timestamp, measured_rate=None):
        """
        Tính toán đầu ra PID.

        Args:
            error: Sai số hiện tại (setpoint - measurement)
            timestamp: Thời gian hiện tại (giây)
            measured_rate: Tốc độ thay đổi đo trực tiếp (ví dụ: từ gyroscope).
                          Nếu cung cấp, sẽ dùng thay vì vi phân error.

        Returns:
            Giá trị đầu ra PID (đã giới hạn)
        """
        if self._prev_time is None:
            self._prev_time = timestamp
            self._prev_error = error
            return 0.0

        dt = timestamp - self._prev_time
        if dt <= 0.0 or dt > 1.0:
            # dt không hợp lệ hoặc quá lớn (bỏ qua)
            self._prev_time = timestamp
            self._prev_error = error
            return self.last_output

        # === Proportional (Tỉ lệ) ===
        self.last_p_term = self.kp * error

        # === Integral (Tích phân) với anti-windup ===
        self._integral += error * dt
        self._integral = max(-self.integral_max,
                             min(self.integral_max, self._integral))
        self.last_i_term = self.ki * self._integral

        # === Derivative (Đạo hàm) với low-pass filter ===
        if measured_rate is not None:
            # Dùng tốc độ đo trực tiếp (gyro) - chính xác hơn
            # error = pitch - target, do đó d(error)/dt = d(pitch)/dt = gyro_y
            raw_derivative = measured_rate
        else:
            # Vi phân error
            raw_derivative = (error - self._prev_error) / dt

        # Lọc thông thấp (low-pass filter)
        self._filtered_derivative = (
            self.alpha * raw_derivative +
            (1.0 - self.alpha) * self._filtered_derivative
        )
        self.last_d_term = self.kd * self._filtered_derivative

        # === Output ===
        output = self.last_p_term + self.last_i_term + self.last_d_term
        output = max(self.output_min, min(self.output_max, output))

        # Lưu trạng thái
        self._prev_error = error
        self._prev_time = timestamp
        self.last_output = output

        return output

    @property
    def integral(self):
        """Giá trị tích phân hiện tại."""
        return self._integral

    def __repr__(self):
        return (f"PIDController(kp={self.kp:.4f}, ki={self.ki:.4f}, "
                f"kd={self.kd:.4f})")
