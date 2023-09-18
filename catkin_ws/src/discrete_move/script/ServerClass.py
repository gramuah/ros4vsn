import rospy
from discrete_move.srv import DiscreteServer, DiscreteServerResponse

import time
from move_robot import MoveRobot


class Server:
    def __init__(self, service_name: str = "discrete_move") -> None:
        self.robot = MoveRobot()
        self.service_name = service_name

        rospy.Service(self.service_name, DiscreteServer, self._callback_action)
        rospy.loginfo("Service up, waiting action")
        rospy.spin()

    def _callback_action(self, req) -> int:

        while self.robot.position_robot is None and not rospy.is_shutdown():
            time.sleep(0.1)

        if req.movement == 'Forward':
            rospy.loginfo("Forward")
            self.robot.move_forward()

        elif req.movement == 'Backward':
            rospy.loginfo("Backward")
            self.robot.move_backward()

        elif req.movement == 'Left':
            rospy.loginfo(f"Left {req.angle}")
            self.robot.turn_left(req.angle)

        elif req.movement == 'Right':
            rospy.loginfo(f"Right {req.angle}")
            self.robot.turn_right(req.angle)

        elif req.movement == 'Stop':
            rospy.loginfo(f"Stop")
            self.robot.stop_robot()

        else:
            rospy.loginfo(f"Client is sending wrong information, robot will stop")
            self.robot.stop_robot()
            return DiscreteServerResponse(False)

        return DiscreteServerResponse(True)
