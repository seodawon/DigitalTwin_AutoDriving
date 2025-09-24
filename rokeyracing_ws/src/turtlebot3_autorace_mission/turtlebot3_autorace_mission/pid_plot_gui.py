import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import matplotlib.pyplot as plt
from collections import deque
import threading

class PIDPlotter(Node):
    def __init__(self):
        super().__init__('pid_plotter')
        self.error_sub = self.create_subscription(Float32, '/pid/error', self.error_cb, 10)
        self.output_sub = self.create_subscription(Float32, '/pid/output', self.output_cb, 10)

        self.error_data = deque(maxlen=200)
        self.output_data = deque(maxlen=200)

    def error_cb(self, msg):
        self.error_data.append(msg.data)

    def output_cb(self, msg):
        self.output_data.append(msg.data)

def run_plotter(node):
    plt.ion()
    fig, ax = plt.subplots()
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.01)
        ax.clear()
        # ax.plot(node.error_data, label='Error')
        ax.plot([e / 10.0 for e in node.error_data], label='Error (x0.1)')
        ax.plot(node.output_data, label='Output')
        ax.legend()
        ax.set_title("PD Tracking")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        plt.pause(0.01)

def main():
    rclpy.init()
    node = PIDPlotter()
    try:
        run_plotter(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
