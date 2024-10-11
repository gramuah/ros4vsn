import math
import rospy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion


class MoveRobot:

    def __init__(self):
        """
        Class to control the robot with parameterized movements
        """
        self._angle_target = 0

        # Robot's update position
        self.position_robot = ()
        self._yaw_robot = .0

        # Declare ROS subscribers and publishers
        self._odom_sub_name = "/odom"
        self._vel_pub_name = "/cmd_vel"

        rospy.Subscriber(self._odom_sub_name, Odometry, self._odom_sub_callback, queue_size=1)
        self._vel_pub = rospy.Publisher(self._vel_pub_name, Twist, queue_size=1)

        # Configurable parameters discrete_move.yaml
        self._linear_velocity = rospy.get_param('/discrete_move/linear_velocity', 0.2)
        self._angular_velocity = rospy.get_param('/discrete_move/angular_velocity', 0.5)
        self._straight_distance = rospy.get_param('/discrete_move/straight_distance', 1)
        self._publish_freq = rospy.get_param('/diff_drive_controller/publish_rate', 10)

        #percentage of the distance the robot will accelerate
        self._acceleration_distance = rospy.get_param('/discrete_move/acceleration_distance',0.3)
        self._initial_velocity_param = rospy.get_param('/discrete_move/initial_velocity_param', 6)
    def move_forward(self, steps=1) -> None:
        """
        Move the robot forward
        :param steps: Number of steps of distances the robot moves forward
        """
        print(f"Move forward {self._straight_distance * steps} meters")
        self._move(steps, 1)
        self.stop_robot()

    def move_backward(self, steps=1) -> None:
        """
        Move the robot backward
        :param steps: Number of steps of distances the robot moves backward
        """
        print(f"Move backward {self._straight_distance * steps} meters")
        self._move(steps, -1)
        self.stop_robot()

    def turn_right(self, angle: int = 90) -> None:
        """
        Move the robot to the right
        :param angle: int angle of rotation to the right
        """
        print(f"Move Right {angle}º")
        self._turn(-1, angle)
        self.stop_robot()

    def turn_left(self, angle: int = 90) -> None:
        """
        Move the robot to the left
        :param angle: int angle of rotation to the left
        """
        print(f"Move Left {angle}º")
        self._turn(1, angle)
        self.stop_robot()

    def stop_robot(self) -> None:
        """
        Publish the stop signal to robot: linear and angular velocity is 0
        """
        r = rospy.Rate(self._publish_freq)
        self._publish_topic(0, 0)
        r.sleep()

    def _move(self, steps: int = 1, forward: int = 1) -> None:
        """
        Set the linear velocity until final distance is reached
        :param steps: Number of steps of distances the robot moves
        :param forward: 1 - Forward or -1 - Backward
        """
        # Calculate the total motion distance
        distance = self._straight_distance * steps
        epsilon = 0.005
        pos_init = self.position_robot

        # Set the velocity forward until distance is reached
        r = rospy.Rate(self._publish_freq)
        velocity = self._linear_velocity
        while not rospy.is_shutdown():
            pos = self.position_robot
            # Calculate euclidean distance between the current position and the final position,
            # then calculate the difference (error) with the total motion distance.
            error = distance - math.hypot(pos[0] - pos_init[0], pos[1] - pos_init[1])

            if error > epsilon:
                #print(velocity)
                if error > distance * (1-self._acceleration_distance):
                    velocity = self._linear_velocity * (distance-error+(self._linear_velocity/self._initial_velocity_param)) * 1/(self._acceleration_distance)
                if error < distance * self._acceleration_distance:
                    velocity = self._linear_velocity * (error +(self._linear_velocity/self._initial_velocity_param)) * 1/(self._acceleration_distance)
                #if(velocity > (self._linear_velocity * 0.3)):
                    #velocity = (error/distance) * self._linear_velocity
                self._publish_topic(forward * velocity, 0)
                r.sleep()
            else:
                break

    def _turn(self, dire: int, turn_angle: int = 90) -> None:
        """
        Set the angular velocity until final angle is reached
        :param dire: 1 -> Left or -1 -> Right
        :param turn_angle: int angle of rotation
        """
        self._update_angle(dire * turn_angle)
        epsilon = .08

        # Turn left
        if dire == 1:
            r = rospy.Rate(self._publish_freq)
            while self._calculate_angle(self._yaw_robot) > epsilon and not rospy.is_shutdown():
                self._publish_topic(0, self._angular_velocity)
                r.sleep()

        # Turn right
        if dire == -1:
            r = rospy.Rate(self._publish_freq)
            while self._calculate_angle(self._yaw_robot) > epsilon and not rospy.is_shutdown():
                self._publish_topic(0, -self._angular_velocity)
                r.sleep()

    def _calculate_angle(self, angle_current: float) -> float:
        """
        Calculate the difference between the angle_current and the self._angle_target
        :param angle_current: float angle (radians)
        :return: difference in radians
        """
        dif = (self._angle_target * math.pi / 180 - angle_current) % (360 * math.pi / 180)
        if dif < math.pi:
            return dif
        else:
            return 2 * math.pi - dif

    def _update_angle(self, angle: int) -> None:
        """
        Calculate the final angle, by adding an angle in degrees
        :param angle: int angle (degrees)
        """
        self._angle_target = (self._angle_target + angle) % 360

    def _publish_topic(self, linear_x: float, angular_z: float) -> None:
        """
        Publish message to topic /odom2
        :param linear_x:
        :param angular_z:
        """
        msg = Twist()

        msg.linear.x = linear_x
        msg.angular.z = angular_z

        self._vel_pub.publish(msg)

    def _odom_sub_callback(self, data) -> None:
        """
        Callback for ROS subscriber /odom
        Get the new update position of robot from topic /odom
        :param data: PoseStamped ROS message
        """
        self.position_robot = data.pose.pose.position.x, data.pose.pose.position.y

        self._yaw_robot = euler_from_quaternion([data.pose.pose.orientation.x,
                                                 data.pose.pose.orientation.y,
                                                 data.pose.pose.orientation.z,
                                                 data.pose.pose.orientation.w])[2] % (2 * math.pi)
