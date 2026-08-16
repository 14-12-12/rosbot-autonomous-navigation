#!/usr/bin/env python3

from __future__ import print_function

import rospy
import math
import actionlib
import sys
import tty
import termios

from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import Float64, String
from std_msgs.msg import Float64

#ALL_WAYPOINTS = [
 #   (0.007460, -0.002640, 0.001297), #start point
  #  (1.859720, -0.040516, 0.003649), # before turn 1
   # (2.458554, -0.398413, -1.589817),#after turn 1
    #(2.449179, -2.121598, -1.562600),#before turn 2
    #(2.050226, -2.640582,  3.066848),#after turn 2
    #(0.591191, -2.627685,  3.114023),#before turn 3
    #(-0.023446, -2.359769,  1.534470),#after turn 3
    #(-0.002930, -1.808338,  1.527711),#traffic light
    #(0.014206, -1.415348,  1.537069),#traffic light
    #(0.547789, -0.959483, -0.008554),#traffic light crossing
    #(1.149264, -1.233288, -0.758574),#after crossing turn full right
    #(1.532759, -1.696802,  0.023985),#half right
    #(2.019488, -1.706339,  0.045186),#before riht
    #(2.493705, -2.181280, -1.605418),#after right
    #(2.182879, -2.727748,  3.082269),
    #(1.472332, -2.691766,  3.132655),
    #(0.977477, -2.148686,  1.551949),
    #(1.058694, -1.212645,  1.554423),
#]
ALL_WAYPOINTS = [
    (0.007460, -0.002640,  0.001297),
    (1.853220, -0.041204, -0.026450),
    (2.454944, -0.489565, -1.382556),
    (2.396480, -1.681115, -1.570707),
    (2.407649, -2.115857, -1.550503),
    (1.849727, -2.730481,  3.116412),
    (1.081811, -2.671891,  3.049426),
    (0.556458, -2.628096,  3.068148),
    (-0.005930, -2.079663,  1.596872),#traffic lgiht
    (0.004803, -1.395114,  1.545308),#after
    (0.543356, -1.055399, -0.105868), #turn right
    (1.012759, -1.280502, -0.601971),#midway
    (1.568161, -1.567699, -0.234900),#straighten
    (1.884225, -1.605907, -0.110143),#straighten
    (2.407649, -2.115857, -1.550503),#truen trigh
    (1.849727, -2.730481,  3.116412),#turned right(maybe add a lower x axis value after if crosses)
    (0.977477, -2.148686,  1.551949),#turned right again
    (1.058694, -0.902645,  1.554423),#parking point
    #(1.078581, -0.311071, 1.594286), #parking 1
    #(1.048352, -0.115406, 1.883994), #parking 2
    #(1.207602, -0.550673, 1.953722), #parking 3
    #(1.372418,-0.998361,1.963741), #parking 4
    #(1.346086,-0.764926,1.454391),#parking 5
]


TRAFFIC_LIGHT_INDEX = 8   # WP20
PARKING_INDEX       = 22   # WP24

XY_TOLERANCE  = 0.35
YAW_TOLERANCE = 2.0


def yaw_to_quaternion(yaw):
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class WaypointNavigator():

    def __init__(self):
        self.current_x   = 0.0
        self.current_y   = 0.0
        self.current_yaw = 0.0
        self.have_pose   = False

        self.pose_sub = rospy.Subscriber(
            '/robot_1/amcl_pose',
            PoseWithCovarianceStamped,
            self.pose_callback,
            queue_size=10
        )

        self.cam_pub = rospy.Publisher(
            '/robot_1/joint1_controller/command',
            Float64, queue_size=1)

        self.nav_complete_pub = rospy.Publisher(
            '/navigation_complete', String, queue_size=1)

        self.move_base_client = actionlib.SimpleActionClient(
            '/robot_1/move_base', MoveBaseAction
        )
        rospy.loginfo("Waiting for move_base action server...")
        self.move_base_client.wait_for_server(rospy.Duration(5.0))
        rospy.loginfo("move_base connected!")

    def pose_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.have_pose = True
        rospy.loginfo_throttle(
            1.0, "Pose -> x=%.3f y=%.3f yaw=%.3f" %
            (self.current_x, self.current_y, self.current_yaw)
        )

    def send_goal(self, x, y, yaw):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "robot_1/map"
        goal.target_pose.header.stamp    = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation = yaw_to_quaternion(yaw)
        print("Sending goal -> x=%.3f y=%.3f yaw=%.3f" % (x, y, yaw))
        self.move_base_client.send_goal(goal)

    def is_reached(self, x, y, yaw):
        dist    = math.sqrt((self.current_x - x) ** 2 + (self.current_y - y) ** 2)
        yaw_err = abs(normalize_angle(self.current_yaw - yaw))
        mb_state = self.move_base_client.get_state()
        mb_done  = mb_state in [3, 4, 5]
        if mb_done or (dist < XY_TOLERANCE and yaw_err < YAW_TOLERANCE):
            rospy.loginfo("Reached! dist=%.2fm yaw_err=%.2frad mb_state=%d" %
                          (dist, yaw_err, mb_state))
            return True
        rospy.loginfo_throttle(
            1.0, "dist=%.2fm yaw_err=%.2frad" % (dist, yaw_err)
        )
        return False

    def run(self):
        print("Waiting for first pose from AMCL...")
        while not self.have_pose and not rospy.is_shutdown():
            rospy.sleep(0.2)
        print("Pose received!")

        rate = rospy.Rate(10)

        for i, (x, y, yaw) in enumerate(ALL_WAYPOINTS):
            if rospy.is_shutdown():
                break

            if i == TRAFFIC_LIGHT_INDEX:
                print("=============================")
                print("TRAFFIC LIGHT REGION REACHED, CHECK LIGHT!")
                print("=============================")


            if i == PARKING_INDEX:
                print("=============================")
                print("PARKING REACHED!")
                print("=============================")

            print("Navigating to waypoint %d/%d: x=%.3f y=%.3f" %
                  (i+1, len(ALL_WAYPOINTS), x, y))

            self.send_goal(x, y, yaw)

            while not rospy.is_shutdown():
                if self.is_reached(x, y, yaw):
                    self.move_base_client.cancel_goal()
                    print("Waypoint %d reached!" % (i+1))

                    if i == 7:  # traffic light zone -- turn camera right
                        rospy.sleep(0.5)
                        self.cam_pub.publish(Float64(0.5))
                        print("Camera turned right")

                    if i == 9:  # after traffic light -- camera back to normal
                        rospy.sleep(0.5)
                        self.cam_pub.publish(Float64(0.0))
                        print("Camera back to normal")

                    break
                rate.sleep()

        print("All waypoints complete!")
        rospy.sleep(0.5)
        self.nav_complete_pub.publish(String('done'))
        print("Published navigation_complete")


if __name__ == '__main__':

    rospy.init_node('waypoint_navigator')

    nav = WaypointNavigator()

    print("=============================")
    print("Press 's' to start navigation")
    print("=============================")

    settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        while True:
            key = sys.stdin.read(1)
            if key == 's' or key == 'S':
                break
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

    print("Starting!")

    try:
        nav.run()
    except KeyboardInterrupt:
        print("Shutting down")
