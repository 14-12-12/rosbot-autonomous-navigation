#!/usr/bin/env python3
from __future__ import print_function

import sys
sys.path.insert(0, '/home/hiwonder/ros_ws/devel/lib/python3/dist-packages')

import rospy
import numpy as np
import cv2
import onnxruntime as rt
import threading

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

MODEL_PATH = '/home/hiwonder/ros_ws/src/data_collect/models/yolov4_2.onnx'

CLASS_NAMES = [
    'lift_speed_limit',   # 0
    'parking',            # 1
    'speed_limit',        # 2
    'stop_sign',          # 3
    'traffic_light',      # 4
]

S_MIN           = 60
V_BRIGHT        = 150
MIN_PIXEL_COUNT = 15

CONF_THRESHOLD = 0.5
NMS_THRESHOLD  = 0.4
INPUT_SIZE     = 416
DEBUG_MODE     = True    # set to False for demo to save CPU

MIN_BOX_AREA_PER_CLASS = {
    0: 2700,   # lift_speed_limit
    1: 2000,   # parking
    2: 1100,    # speed_limit
    3: 1500,   # stop_sign
    4: 18000,   # traffic_light
}

BOX_COLOURS = [
    (255, 0,   0),    # lift_speed_limit -- blue
    (0,   255, 0),    # parking          -- green
    (0,   0,   255),  # speed_limit      -- red
    (255, 255, 0),    # stop_sign        -- cyan
    (0,   255, 255),  # traffic_light    -- yellow
]


class SignDetectionNode(object):

    def __init__(self):
        rospy.init_node('sign_detection_node', anonymous=False)
        rospy.loginfo('Loading ONNX model from: %s' % MODEL_PATH)

        self.session      = rt.InferenceSession(MODEL_PATH, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider','CPUExecutionProvider'])
        self.input_name   = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        rospy.loginfo('Model loaded OK. Input name: %s' % self.input_name)
        rospy.loginfo('Output layers: %s' % str(self.output_names))

        self.bridge = CvBridge()

        self.sign_pub  = rospy.Publisher('/detected_sign',       String, queue_size=10)
        self.light_pub = rospy.Publisher('/traffic_light_state', String, queue_size=10)

        # ── FIX 5 — shared latest frame with thread lock ──────
        self.latest_frame = None
        self.frame_lock   = threading.Lock()

        # ── FIX 2 — processing flag to drop stale frames ──────
        self.processing   = False

        self.last_states    = []
        self.CONFIRM_FRAMES = 1

        # Cooldown per class — seconds before same sign published again
        self.SIGN_COOLDOWN = {
            0: 360.0,    # lift_speed_limit
            1: 360.0,   # parking
            2: 360.0,    # speed_limit
            3: 360.0,    # stop_sign
            4: 0.5,    # traffic_light — needs frequent updates
        }
        self.last_published_time = {}  # class_id -> last publish time

        # Start background processing thread
        self.process_thread        = threading.Thread(target=self.process_loop)
        self.process_thread.daemon = True
        self.process_thread.start()
        rospy.loginfo('Processing thread started')

        # Subscribe to camera — callback only stores frame, never blocks
        self.sub = rospy.Subscriber(
            '/robot_1/depth_cam/rgb/image_raw',
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )

        rospy.loginfo('sign_detection_node ready.')

    # ----------------------------------------------------------
    # FIX 5 — image_callback only stores latest frame
    # Never runs inference here — returns immediately
    # ----------------------------------------------------------

    def image_callback(self, msg):
        """
        FIX 5: Just store the latest frame.
        Never block here — return immediately so camera
        does not fall behind.
        """
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.frame_lock:
                self.latest_frame = frame  # always overwrite with latest
        except Exception as e:
            rospy.logerr('CvBridge error: %s' % str(e))

    # ----------------------------------------------------------
    # FIX 5 — background thread runs detection at controlled rate
    # ----------------------------------------------------------

    def process_loop(self):
        """
        FIX 5: Run detection in background thread at 5Hz.
        Always takes the latest frame — never processes stale ones.
        """
        rate = rospy.Rate(5)  # 5Hz — adjust if needed
        while not rospy.is_shutdown():
            frame = None

            # Take latest frame
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame             = self.latest_frame.copy()
                    self.latest_frame = None  # clear so we don't reprocess

            if frame is not None:
                # FIX 2 — skip if already processing
                if not self.processing:
                    self.processing = True
                    try:
                        self.run_detection(frame)
                    except Exception as e:
                        rospy.logerr('Detection error: %s' % str(e))
                    finally:
                        self.processing = False

            rate.sleep()

    # ----------------------------------------------------------
    # Detection logic — moved from image_callback
    # ----------------------------------------------------------

    def run_detection(self, frame):
        """Run full detection pipeline on a single frame."""
        orig_h, orig_w = frame.shape[:2]
        inp = self.preprocess(frame)

        try:
            outputs = self.session.run(self.output_names, {self.input_name: inp})
        except Exception as e:
            rospy.logerr('Inference error: %s' % str(e))
            return

        detections = self.postprocess(outputs, orig_h, orig_w)
        best       = self.pick_best_detection(detections)

        if DEBUG_MODE:
            for d in detections:
                min_area = self.get_min_area(d['class_id'])
                if best is not None and d is best:
                    status = 'PUBLISHED'
                elif d['area'] < min_area:
                    status = 'ignored - too small (threshold=%d)' % min_area
                else:
                    status = 'ignored - not best'
                print('[DEBUG] %-20s conf=%.2f  area=%-6d  (%s)' % (
                    CLASS_NAMES[d['class_id']], d['confidence'], d['area'], status))

        if best is not None:
            sign_name = CLASS_NAMES[best['class_id']]
            class_id  = best['class_id']
            now       = rospy.Time.now().to_sec()

            # Check cooldown
            last_time   = self.last_published_time.get(class_id, 0.0)
            cooldown    = self.SIGN_COOLDOWN.get(class_id, 5.0)
            on_cooldown = (now - last_time) < cooldown

            if sign_name == 'traffic_light':
                # Traffic light always processes — no cooldown on detection
                bbox         = (best['x'], best['y'], best['w'], best['h'])
                light_colour = self.detect_light_colour(frame, bbox)

                self.last_states.append(light_colour)
                if len(self.last_states) > self.CONFIRM_FRAMES:
                    self.last_states.pop(0)

                if self.last_states.count(light_colour) >= self.CONFIRM_FRAMES:
                    self.light_pub.publish(light_colour)
                    if DEBUG_MODE:
                        print(">>> LIGHT PUBLISHED: " + light_colour)

                self.sign_pub.publish(String('traffic_light'))

                x, y = bbox[0], bbox[1]
                cv2.putText(frame, light_colour, (x, max(y - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            elif not on_cooldown:
                # Publish sign and start cooldown
                self.sign_pub.publish(String(sign_name))
                self.last_published_time[class_id] = now
                rospy.loginfo('Published sign: %s' % sign_name)

            else:
                # On cooldown — show in debug but do not publish
                if DEBUG_MODE:
                    print('[COOLDOWN] %s — %.1fs remaining' % (
                        sign_name, cooldown - (now - last_time)))

        else:
            self.light_pub.publish('NONE')

        frame = self.draw_detections(frame, detections, best)
        cv2.imshow('Sign Detection', frame)
        cv2.waitKey(1)

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def get_min_area(self, class_id):
        return MIN_BOX_AREA_PER_CLASS.get(class_id, 2000)

    def preprocess(self, frame):
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def postprocess(self, outputs, orig_h, orig_w):
        boxes_out = outputs[0][0]
        confs_out = outputs[1][0]

        boxes       = []
        confidences = []
        class_ids   = []

        for i in range(len(boxes_out)):
            class_scores = confs_out[i]
            class_id     = int(np.argmax(class_scores))
            confidence   = float(class_scores[class_id])

            if confidence < CONF_THRESHOLD:
                continue

            box = boxes_out[i][0]
            x1  = float(box[0]) * orig_w
            y1  = float(box[1]) * orig_h
            x2  = float(box[2]) * orig_w
            y2  = float(box[3]) * orig_h

            x = int(x1)
            y = int(y1)
            w = int(x2 - x1)
            h = int(y2 - y1)

            boxes.append([x, y, w, h])
            confidences.append(confidence)
            class_ids.append(class_id)

        if len(boxes) == 0:
            return []

        indices    = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        detections = []

        if len(indices) > 0:
            for i in indices:
                if isinstance(i, (list, np.ndarray)):
                    i = i[0]
                x, y, w, h = boxes[i]
                detections.append({
                    'class_id':   class_ids[i],
                    'confidence': confidences[i],
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'area': w * h,
                })

        return detections

    def pick_best_detection(self, detections):
        valid = [
            d for d in detections
            if d['area'] >= self.get_min_area(d['class_id'])
        ]
        if not valid:
            return None
        return max(valid, key=lambda d: d['area'])

    def detect_light_colour(self, frame, bbox):
        x, y, w, h = bbox
        ih, iw     = frame.shape[:2]
        x          = max(0, x)
        y          = max(0, y)
        x2         = min(iw, x + w)
        y2         = min(ih, y + h)

        roi = frame[y:y2, x:x2]
        if roi.size == 0:
            return "NONE"

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0,   S_MIN, V_BRIGHT]), np.array([15,  255, 255])),
            cv2.inRange(hsv, np.array([160, S_MIN, V_BRIGHT]), np.array([180, 255, 255]))
        )
        yellow_mask = cv2.inRange(hsv, np.array([15, S_MIN, V_BRIGHT]), np.array([40,  255, 255]))
        green_mask  = cv2.inRange(hsv, np.array([40, S_MIN, V_BRIGHT]), np.array([85,  255, 255]))

        red_count    = int(np.sum(red_mask    > 0))
        yellow_count = int(np.sum(yellow_mask > 0))
        green_count  = int(np.sum(green_mask  > 0))

        if DEBUG_MODE:
            print("HSV+bright: red=%d yellow=%d green=%d" % (
                red_count, yellow_count, green_count))

        best_count = max(red_count, yellow_count, green_count)
        if best_count < MIN_PIXEL_COUNT:
            return "NONE"

        if red_count >= yellow_count and red_count >= green_count:
            return "RED"
        elif yellow_count >= red_count and yellow_count >= green_count:
            return "YELLOW"
        else:
            return "GREEN"

    def draw_detections(self, frame, detections, best):
        for d in detections:
            x, y, w, h = d['x'], d['y'], d['w'], d['h']
            cid        = d['class_id']
            label      = '%s %.2f' % (CLASS_NAMES[cid], d['confidence'])
            colour     = BOX_COLOURS[cid % len(BOX_COLOURS)]

            if d['area'] < self.get_min_area(cid):
                colour = (128, 128, 128)
                label  = label + ' (far)'

            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(frame, label, (x, max(y - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

        if best is not None:
            x, y, w, h = best['x'], best['y'], best['w'], best['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
            cv2.putText(frame, 'PUBLISHING', (x, max(y - 28, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame


if __name__ == '__main__':
    try:
        node = SignDetectionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
