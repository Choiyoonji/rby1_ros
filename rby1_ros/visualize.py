import rclpy
from rclpy.node import Node
import numpy as np
import open3d as o3d
from rby1_interfaces.msg import State, Command

class VisualizeStateVsCommand(Node):
    """
    /rby1/state 와 /control/command 를 구독하여 3D 뷰로 현재 포즈(원)와 목표 포즈(십자) 비교.
    """
    def __init__(self):
        super().__init__('visualize_state_vs_command')
        self.state_sub = self.create_subscription(State, '/rby1/state', self.state_cb, 10)
        self.cmd_sub = self.create_subscription(Command, '/control/command', self.cmd_cb, 10)
        self.timer = self.create_timer(1.0/20.0, self.update_display)  # 20 Hz

        self.current = {'torso': None, 'right': None, 'left': None}
        self.desired = {'torso': None, 'right': None, 'left': None}
        self.is_active = False

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("rby_compare", width=800, height=600)

    def state_cb(self, msg: State):
        try:
            self.current['torso'] = np.array(msg.torso_ee_pos.position.data, dtype=float)
            self.current['right'] = np.array(msg.right_ee_pos.position.data, dtype=float)
            self.current['left']  = np.array(msg.left_ee_pos.position.data, dtype=float)
        except Exception:
            pass

    def cmd_cb(self, msg: Command):
        try:
            self.desired['torso'] = np.array(msg.desired_torso_ee_pos.position.data, dtype=float)
            self.desired['right'] = np.array(msg.desired_right_ee_pos.position.data, dtype=float)
            self.desired['left']  = np.array(msg.desired_left_ee_pos.position.data, dtype=float)
            self.is_active = True
        except Exception:
            self.is_active = False

    def update_display(self):
        self.vis.clear_geometries()

        # Draw current positions
        if self.current['torso'] is not None:
            torso_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            torso_point.paint_uniform_color([1, 0.8, 0])  # yellow
            torso_point.translate(self.current['torso'])
            self.vis.add_geometry(torso_point)

        if self.current['right'] is not None:
            right_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            right_point.paint_uniform_color([1, 0, 0])  # red
            right_point.translate(self.current['right'])
            self.vis.add_geometry(right_point)

        if self.current['left'] is not None:
            left_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            left_point.paint_uniform_color([0, 1, 0])  # green
            left_point.translate(self.current['left'])
            self.vis.add_geometry(left_point)

        # Draw desired positions
        if self.desired['torso'] is not None:
            torso_cross = self.create_cross(self.desired['torso'], color=[1, 0.8, 0])
            self.vis.add_geometry(torso_cross)

        if self.desired['right'] is not None:
            right_cross = self.create_cross(self.desired['right'], color=[1, 0, 0])
            self.vis.add_geometry(right_cross)

        if self.desired['left'] is not None:
            left_cross = self.create_cross(self.desired['left'], color=[0, 1, 0])
            self.vis.add_geometry(left_cross)

        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()

    def create_cross(self, position, size=0.1, color=[1, 0, 0]):
        lines = []
        for dx in [-size, size]:
            lines.append([position + np.array([dx, 0, 0]), position + np.array([0, dx, 0])])
            lines.append([position + np.array([0, dx, 0]), position + np.array([dx, 0, 0])])
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.array(lines).reshape(-1, 3))
        line_set.lines = o3d.utility.Vector2iVector(np.array([[i, i + 1] for i in range(0, len(lines) * 2, 2)]))
        line_set.paint_uniform_color(color)
        return line_set

def main(args=None):
    rclpy.init(args=args)
    node = VisualizeStateVsCommand()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    node.vis.destroy_window()
    rclpy.shutdown()

if __name__ == '__main__':
    main()