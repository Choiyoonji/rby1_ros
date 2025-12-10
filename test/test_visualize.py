import unittest
from rclpy import init, shutdown
from rclpy.node import Node
from rby1_ros.visualize import VisualizeStateVsCommand
from rby1_interfaces.msg import StateRBY1, Command

class TestVisualizeStateVsCommand(unittest.TestCase):
    def setUp(self):
        init()
        self.node = VisualizeStateVsCommand()

    def tearDown(self):
        self.node.destroy_node()
        shutdown()

    def test_initialization(self):
        self.assertIsNotNone(self.node.state_sub)
        self.assertIsNotNone(self.node.cmd_sub)
        self.assertEqual(self.node.is_active, False)

    def test_state_callback(self):
        test_state = StateRBY1()
        test_state.torso_ee_pos.position.data = [1.0, 2.0, 0.0]
        test_state.right_ee_pos.position.data = [1.5, 2.5, 0.0]
        test_state.left_ee_pos.position.data = [0.5, 1.5, 0.0]

        self.node.state_cb(test_state)

        self.assertTrue((self.node.current['torso'] == [1.0, 2.0, 0.0]).all())
        self.assertTrue((self.node.current['right'] == [1.5, 2.5, 0.0]).all())
        self.assertTrue((self.node.current['left'] == [0.5, 1.5, 0.0]).all())

    def test_command_callback(self):
        test_command = Command()
        test_command.desired_torso_ee_pos.position.data = [1.0, 2.0, 0.0]
        test_command.desired_right_ee_pos.position.data = [1.5, 2.5, 0.0]
        test_command.desired_left_ee_pos.position.data = [0.5, 1.5, 0.0]

        self.node.cmd_cb(test_command)

        self.assertTrue((self.node.desired['torso'] == [1.0, 2.0, 0.0]).all())
        self.assertTrue((self.node.desired['right'] == [1.5, 2.5, 0.0]).all())
        self.assertTrue((self.node.desired['left'] == [0.5, 1.5, 0.0]).all())
        self.assertTrue(self.node.is_active)

    def test_world_to_img(self):
        pos = [1.0, 1.0, 0.0]
        img_pos = self.node.world_to_img(pos)
        expected_pos = (self.node.origin[0] + 1.0 * self.node.scale, 
                        self.node.origin[1] - 1.0 * self.node.scale)
        self.assertEqual(img_pos, (int(expected_pos[0]), int(expected_pos[1])))

if __name__ == '__main__':
    unittest.main()