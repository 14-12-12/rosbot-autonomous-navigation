#!/usr/bin/env python3
from __future__ import print_function

import sys
sys.path.insert(0, '/home/hiwonder/ros_ws/devel/lib/python3/dist-packages')

import rospy
import os
import math
import actionlib

from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from move_base_msgs.msg import MoveBaseAction

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from parking_policy import ParkingPolicy

# ── Speeds ────────────────────────────────────────────────────
SPEED_SLOW        = 0.08
SPEED_STOP        = 0.00

# ── Timing ────────────────────────────────────────────────────
STOP_SIGN_WAIT      = 3.0    # seconds to wait at stop sign
PARKING_SETTLE_TIME = 1.0    # seconds to settle before recording origin
STOP_SIGN_COOLDOWN  = 10.0   # seconds to cooldown

# ── Parking ───────────────────────────────────────────────────
MAX_PARKING_STEPS  = 320
PARKING_STOP_COUNT = 5       # consecutive near-zero cmds = parked

# ── Topics ────────────────────────────────────────────────────
SIGN_TOPIC        = '/detected_sign'
LIGHT_TOPIC       = '/traffic_light_state'
ODOM_TOPIC        = '/robot_1/odom'
CMD_OUT_TOPIC     = '/robot_1/cmd_vel'
MOVE_BASE_TOPIC   = '/robot_1/move_base'

# ── States ────────────────────────────────────────────────────
LANE_FOLLOWING = 'LANE_FOLLOWING'
STOP_RED       = 'STOP_RED'
SLOW_YELLOW    = 'SLOW_YELLOW'
STOP_SIGN      = 'STOP_SIGN'
SPEED_LIMIT    = 'SPEED_LIMIT'
PARKING_SETTLE = 'PARKING_SETTLE'
PARKING        = 'PARKING'

OVERRIDE_STATES = (PARKING_SETTLE, PARKING)


class StateMachine(object):

    def __init__(self):
        # FIX 1 — corrected node name (was 'state_elf.seemachine_node')
        rospy.init_node('state_machine_node', anonymous=False)

        self.state                = LANE_FOLLOWING
        self.stop_sign_start      = None
        self.red_light_start      = None
        self.last_stop_sign_time  = 0.0 
        self.parking_done         = False
        self.parking_steps        = 0
        self.parking_stop_counter = 0
        self.settle_start         = None

        # Speed limit delay
        self.pending_speed_limit  = False
        self.speed_limit_start_x  = 0.0
        self.speed_limit_start_y  = 0.0
        self.DELAY_DISTANCE       = 1.4   # metres to travel before slowing

        # Stop sign delay
        self.pending_stop_sign    = False
        self.stop_sign_start_x    = 0.0
        self.stop_sign_start_y    = 0.0
        self.STOP_SIGN_DELAY_DIST = 0.95   # metres to travel before stopping

        # Lift speed limit delay
        self.pending_lift_speed   = False
        self.lift_speed_start_x   = 0.0
        self.lift_speed_start_y   = 0.0
        self.LIFT_SPEED_DELAY_DIST = 0.4  # metres to travel before lifting

        # SLOW_YELLOW timeout — resumes if GREEN never seen
        self.slow_yellow_start   = None
        self.SLOW_YELLOW_TIMEOUT = 2.0   # seconds before auto-resuming

        self.latest_move_base_cmd = Twist()

        self.x     = 0.0
        self.y     = 0.0
        self.theta = 0.0

        self.origin_x     = 0.0
        self.origin_y     = 0.0
        self.origin_theta = 0.0

        self.parking_policy = ParkingPolicy()

        self.move_base_client = actionlib.SimpleActionClient(
            MOVE_BASE_TOPIC, MoveBaseAction)
        rospy.loginfo('Waiting for move_base...')
        connected = self.move_base_client.wait_for_server(
            rospy.Duration(5.0))
        if connected:
            rospy.loginfo('move_base connected')
        else:
            rospy.logwarn('move_base not available — goal cancellation disabled')

        rospy.Subscriber(SIGN_TOPIC,             String,   self.cb_sign)
        rospy.Subscriber(LIGHT_TOPIC,            String,   self.cb_light)
        rospy.Subscriber(ODOM_TOPIC,             Odometry, self.cb_odom)
        rospy.Subscriber(CMD_OUT_TOPIC,          Twist,    self.cb_move_base_cmd)
        rospy.Subscriber('/navigation_complete', String,   self.cb_nav_complete)

        self.cmd_pub = rospy.Publisher(CMD_OUT_TOPIC, Twist, queue_size=1)

        self.rate = rospy.Rate(20)

        rospy.loginfo('StateMachine ready')
        rospy.loginfo('  Signs:  %s' % SIGN_TOPIC)
        rospy.loginfo('  Lights: %s' % LIGHT_TOPIC)
        rospy.loginfo('  CmdOut: %s' % CMD_OUT_TOPIC)
        rospy.loginfo('State: LANE_FOLLOWING (move_base in control)')

    # ── Callbacks ─────────────────────────────────────────────

    def cb_odom(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.theta = math.atan2(siny_cosp, cosy_cosp)

    def cb_move_base_cmd(self, msg):
        if self.state == LANE_FOLLOWING:
            self.latest_move_base_cmd = msg

    def cb_light(self, msg):
        if self.parking_done:
            return

        light = msg.data.upper().strip()

        if light == 'RED':
            self.slow_yellow_start = None
            self.set_state(STOP_RED)

        elif light == 'YELLOW':
            if self.state not in (STOP_RED, STOP_SIGN, PARKING_SETTLE, PARKING):
                self.set_state(SLOW_YELLOW)
                # Start timer only when first entering SLOW_YELLOW
                if self.slow_yellow_start is None:
                    self.slow_yellow_start = rospy.Time.now()

        elif light == 'GREEN':
            self.slow_yellow_start = None
            if self.state in (STOP_RED, SLOW_YELLOW):
                self.set_state(LANE_FOLLOWING)

    def cb_sign(self, msg):
        if self.parking_done:
            return

        sign = msg.data.lower().strip()

        if sign == 'stop_sign':
            now = rospy.Time.now().to_sec()
            if (self.state == LANE_FOLLOWING
                    and not self.pending_stop_sign
                    and (now - self.last_stop_sign_time) > STOP_SIGN_COOLDOWN):
                rospy.loginfo(
                    'Stop sign spotted — will stop after %.1fm' % self.STOP_SIGN_DELAY_DIST)
                self.pending_stop_sign   = True
                self.stop_sign_start_x   = self.x
                self.stop_sign_start_y   = self.y
                self.last_stop_sign_time = now

        elif sign == 'speed_limit':
            # FIX 3 — record position but do NOT slow down yet
            # Robot will slow down after travelling DELAY_DISTANCE metres
            if self.state == LANE_FOLLOWING and not self.pending_speed_limit:
                rospy.loginfo(
                    'Speed limit spotted — will slow after %.1fm' % self.DELAY_DISTANCE)
                self.pending_speed_limit = True
                self.speed_limit_start_x = self.x
                self.speed_limit_start_y = self.y

        elif sign == 'lift_speed_limit':
            # Record position — lift speed after travelling LIFT_SPEED_DELAY_DIST
            if self.state == SPEED_LIMIT and not self.pending_lift_speed:
                rospy.loginfo(
                    'Lift speed limit spotted — will lift after %.1fm' % self.LIFT_SPEED_DELAY_DIST)
                self.pending_lift_speed = True
                self.lift_speed_start_x = self.x
                self.lift_speed_start_y = self.y
            # Also clear pending speed limit flag regardless
            self.pending_speed_limit = False

        elif sign == 'parking':
            # parking is triggered by navigation_complete, not sign
            pass

        elif sign == 'traffic_light':
            pass

    def cb_nav_complete(self, msg):
        if not self.parking_done and self.state not in (PARKING_SETTLE, PARKING):
            rospy.loginfo('Navigation complete -- starting parking')
            self.settle_start         = rospy.Time.now()
            self.parking_steps        = 0
            self.parking_stop_counter = 0
            self.set_state(PARKING_SETTLE)

    # ── State helpers ─────────────────────────────────────────

    def set_state(self, new_state):
        if self.state != new_state:
            rospy.loginfo('FSM: %s --> %s' % (self.state, new_state))
            self.state = new_state

            if new_state in OVERRIDE_STATES:
                try:
                    self.move_base_client.cancel_all_goals()
                    rospy.loginfo('move_base goals cancelled')
                except Exception as e:
                    rospy.logwarn('Could not cancel move_base goals: %s' % str(e))

    def publish_cmd(self, linear_x, angular_z):
        cmd = Twist()
        cmd.linear.x  = linear_x
        cmd.angular.z = angular_z
        self.cmd_pub.publish(cmd)

    def publish_stop(self):
        self.publish_cmd(SPEED_STOP, 0.0)

    def publish_slow(self):
        cmd = Twist()
        cmd.linear.x  = SPEED_SLOW
        cmd.angular.z = self.latest_move_base_cmd.angular.z * 0.5
        self.cmd_pub.publish(cmd)

    # ── Parking helpers ───────────────────────────────────────

    def to_local_frame(self, gx, gy, gtheta):
        dx    = gx - self.origin_x
        dy    = gy - self.origin_y
        cos_o = math.cos(-self.origin_theta)
        sin_o = math.sin(-self.origin_theta)
        lx    = dx * cos_o - dy * sin_o
        ly    = dx * sin_o + dy * cos_o
        lt    = math.atan2(
            math.sin(gtheta - self.origin_theta),
            math.cos(gtheta - self.origin_theta))
        return lx, ly, lt

    def execute_parking(self):
        if self.parking_done:
            self.publish_stop()
            return

        if self.parking_steps >= MAX_PARKING_STEPS:
            rospy.logwarn('Parking timeout!')
            self.publish_stop()
            self.parking_done = True
            return

        rel_x, rel_y, rel_theta = self.to_local_frame(
            self.x, self.y, self.theta)

        linear_x, angular_z = self.parking_policy.predict(
            rel_x, rel_y, rel_theta)

        rospy.loginfo(
            'Parking step %d: rel=(%.3f,%.3f,%.3f) cmd=(%.3f,%.3f)' % (
                self.parking_steps,
                rel_x, rel_y, rel_theta,
                linear_x, angular_z))

        self.publish_cmd(linear_x, angular_z)
        self.parking_steps += 1

        if abs(linear_x) < 0.01 and abs(angular_z) < 0.01:
            self.parking_stop_counter += 1
        else:
            self.parking_stop_counter = 0

        if self.parking_stop_counter >= PARKING_STOP_COUNT:
            self.publish_stop()
            self.parking_done = True
            rospy.loginfo('Parking complete!')

    # ── Main loop ─────────────────────────────────────────────

    def run(self):
        while not rospy.is_shutdown():

            # Speed limit delay check
            if self.pending_speed_limit and self.state == LANE_FOLLOWING:
                dist_moved = math.sqrt(
                    (self.x - self.speed_limit_start_x) ** 2 +
                    (self.y - self.speed_limit_start_y) ** 2
                )
                if dist_moved >= self.DELAY_DISTANCE:
                    rospy.loginfo(
                        'Delay distance %.1fm reached — slowing down' % self.DELAY_DISTANCE)
                    self.set_state(SPEED_LIMIT)
                    self.pending_speed_limit = False

            # Lift speed limit delay check
            if self.pending_lift_speed and self.state == SPEED_LIMIT:
                dist_moved = math.sqrt(
                    (self.x - self.lift_speed_start_x) ** 2 +
                    (self.y - self.lift_speed_start_y) ** 2
                )
                if dist_moved >= self.LIFT_SPEED_DELAY_DIST:
                    rospy.loginfo(
                        'Lift speed delay %.1fm reached — resuming normal speed' % self.LIFT_SPEED_DELAY_DIST)
                    self.pending_lift_speed = False
                    self.set_state(LANE_FOLLOWING)

            # Stop sign delay check
            if self.pending_stop_sign and self.state == LANE_FOLLOWING:
                dist_moved = math.sqrt(
                    (self.x - self.stop_sign_start_x) ** 2 +
                    (self.y - self.stop_sign_start_y) ** 2
                )
                if dist_moved >= self.STOP_SIGN_DELAY_DIST:
                    rospy.loginfo(
                        'Stop sign delay %.1fm reached — stopping' % self.STOP_SIGN_DELAY_DIST)
                    self.set_state(STOP_SIGN)
                    self.stop_sign_start   = rospy.Time.now()
                    self.pending_stop_sign = False

            if self.state == LANE_FOLLOWING:
                pass

            elif self.state == STOP_RED:
                self.publish_stop()
                if self.red_light_start is None:
                    self.red_light_start = rospy.Time.now()
                elapsed = (rospy.Time.now() - self.red_light_start).to_sec()
                if elapsed >= 6.0:
                    rospy.loginfo('Red light wait done — resuming')
                    self.red_light_start = None
                    self.set_state(LANE_FOLLOWING)

            elif self.state == SLOW_YELLOW:
                self.publish_slow()
                # Timeout — resume if GREEN never seen after X seconds
                if self.slow_yellow_start is not None:
                    elapsed = (rospy.Time.now() - self.slow_yellow_start).to_sec()
                    if elapsed >= self.SLOW_YELLOW_TIMEOUT:
                        rospy.loginfo('SLOW_YELLOW timeout — resuming')
                        self.slow_yellow_start = None
                        self.set_state(LANE_FOLLOWING)

            elif self.state == STOP_SIGN:
                self.publish_stop()
                if self.stop_sign_start is not None:
                    elapsed = (rospy.Time.now() - self.stop_sign_start).to_sec()
                    if elapsed >= STOP_SIGN_WAIT:
                        rospy.loginfo('Stop sign wait done — resuming')
                        self.set_state(LANE_FOLLOWING)
                        self.stop_sign_start = None

            elif self.state == SPEED_LIMIT:
                self.publish_slow()

            elif self.state == PARKING_SETTLE:
                self.publish_stop()
                if self.settle_start is not None:
                    elapsed = (rospy.Time.now() - self.settle_start).to_sec()
                    if elapsed >= PARKING_SETTLE_TIME:
                        self.origin_x     = self.x
                        self.origin_y     = self.y
                        self.origin_theta = self.theta
                        rospy.loginfo(
                            'Parking origin: x=%.3f y=%.3f theta=%.3f' % (
                                self.origin_x,
                                self.origin_y,
                                self.origin_theta))
                        self.set_state(PARKING)

            elif self.state == PARKING:
                self.execute_parking()

            self.rate.sleep()


if __name__ == '__main__':
    try:
        sm = StateMachine()
        sm.run()
    except rospy.ROSInterruptException:
        pass
