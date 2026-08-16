#!/usr/bin/env python3
from __future__ import print_function

import rospy
import numpy as np
import cv2
import onnxruntime as rt

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

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
INPUT_SIZE = 416
DEBUG_MODE = True

# ============================================================
# HSV + Brightness 参数 -- 现场调这里
# ============================================================
# 饱和度最低值
S_MIN = 60
# 亮度筛选阈值 -- 只看亮度高于这个的像素
V_BRIGHT = 150
# 最少需要多少个亮色像素才算检测到
MIN_PIXEL_COUNT = 15
# ============================================================

MIN_BOX_AREA_PER_CLASS = {
    0: 5000,
    1: 2000,
    2: 800,
    3: 3500,
    4: 14000,
}

BOX_COLOURS = [
    (255, 0,   0),
    (0,   255, 0),
    (0,   0,   255),
    (255, 255, 0),
    (0,   255, 255),
]


class SignDetectionNode(object):

    def __init__(self):
        rospy.init_node('sign_detection_node', anonymous=False)
        rospy.loginfo('Loading ONNX model from: %s' % MODEL_PATH)

        self.session = rt.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        rospy.loginfo('Model loaded OK.')

        self.bridge = CvBridge()
        self.sign_pub  = rospy.Publisher('/detected_sign',       String, queue_size=10)
        self.light_pub = rospy.Publisher('/traffic_light_state', String, queue_size=10)

        self.sub = rospy.Subscriber(
            '/robot_1/depth_cam/rgb/image_raw',
            Image, self.image_callback,
            queue_size=1, buff_size=2**24
        )

        self.last_states    = []
        self.CONFIRM_FRAMES = 2
        rospy.loginfo('sign_detection_node (HSV+brightness mode) ready.')

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
        boxes, confidences, class_ids = [], [], []

        for i in range(len(boxes_out)):
            class_scores = confs_out[i]
            class_id     = int(np.argmax(class_scores))
            confidence   = float(class_scores[class_id])
            if confidence < CONF_THRESHOLD:
                continue
            box = boxes_out[i][0]
            x1 = float(box[0]) * orig_w
            y1 = float(box[1]) * orig_h
            x2 = float(box[2]) * orig_w
            y2 = float(box[3]) * orig_h
            boxes.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
            confidences.append(confidence)
            class_ids.append(class_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        detections = []
        if len(indices) > 0:
            for i in indices:
                if isinstance(i, (list, np.ndarray)):
                    i = i[0]
                x, y, w, h = boxes[i]
                detections.append({
                    'class_id': class_ids[i], 'confidence': confidences[i],
                    'x': x, 'y': y, 'w': w, 'h': h, 'area': w * h,
                })
        return detections

    def pick_best_detection(self, detections):
        valid = [d for d in detections if d['area'] >= self.get_min_area(d['class_id'])]
        if not valid:
            return None
        return max(valid, key=lambda d: d['area'])

    def detect_light_colour(self, frame, bbox):
        x, y, w, h = bbox
        ih, iw = frame.shape[:2]
        x  = max(0, x)
        y  = max(0, y)
        x2 = min(iw, x + w)
        y2 = min(ih, y + h)

        roi = frame[y:y2, x:x2]
        if roi.size == 0:
            return "NONE"

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_ch = hsv[:,:,0]
        s_ch = hsv[:,:,1]
        v_ch = hsv[:,:,2]

        # 只看高亮度像素
        bright_mask = v_ch > V_BRIGHT

        # 颜色mask（加饱和度过滤）
        s_mask = s_ch > S_MIN

        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0,   S_MIN, V_BRIGHT]), np.array([15,  255, 255])),
            cv2.inRange(hsv, np.array([160, S_MIN, V_BRIGHT]), np.array([180, 255, 255]))
        )
        yellow_mask = cv2.inRange(hsv, np.array([15, S_MIN, V_BRIGHT]), np.array([40, 255, 255]))
        green_mask  = cv2.inRange(hsv, np.array([40, S_MIN, V_BRIGHT]), np.array([85, 255, 255]))

        # 过曝的灯泡饱和度低但亮度极高，单独处理
        overexposed = (v_ch > 220) & (s_ch < 50)
        overexposed_mask = overexposed.astype(np.uint8) * 255

        red_count    = int(np.sum(red_mask    > 0))
        yellow_count = int(np.sum(yellow_mask > 0))
        green_count  = int(np.sum(green_mask  > 0))

        print("HSV+bright: red=%d yellow=%d green=%d overexposed=%d" % (
            red_count, yellow_count, green_count, int(np.sum(overexposed))))

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
            cid    = d['class_id']
            label  = '%s %.2f' % (CLASS_NAMES[cid], d['confidence'])
            colour = BOX_COLOURS[cid % len(BOX_COLOURS)]
            if d['area'] < self.get_min_area(cid):
                colour = (128, 128, 128)
                label  = label + ' (far)'
            cv2.rectangle(frame, (x, y), (x+w, y+h), colour, 2)
            cv2.putText(frame, label, (x, max(y-8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
        if best is not None:
            x, y, w, h = best['x'], best['y'], best['w'], best['h']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 3)
        return frame

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr('CvBridge error: %s' % str(e))
            return

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
                status = 'PUBLISHED' if (best is not None and d is best) else \
                         ('ignored - too small (threshold=%d)' % min_area if d['area'] < min_area else 'ignored - not best')
                print('[DEBUG] %-20s conf=%.2f  area=%-6d  (%s)' % (
                    CLASS_NAMES[d['class_id']], d['confidence'], d['area'], status))

        if best is not None:
            sign_name = CLASS_NAMES[best['class_id']]
            if sign_name == 'traffic_light':
                bbox         = (best['x'], best['y'], best['w'], best['h'])
                light_colour = self.detect_light_colour(frame, bbox)

                self.last_states.append(light_colour)
                if len(self.last_states) > self.CONFIRM_FRAMES:
                    self.last_states.pop(0)
                if self.last_states.count(light_colour) >= self.CONFIRM_FRAMES:
                    self.light_pub.publish(light_colour)
                    print(">>> LIGHT PUBLISHED: " + light_colour)

                self.sign_pub.publish(String('traffic_light'))
                x, y = bbox[0], bbox[1]
                cv2.putText(frame, light_colour, (x, max(y-10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                self.sign_pub.publish(String(sign_name))
                rospy.loginfo('Published sign: %s' % sign_name)
        else:
            self.light_pub.publish('NONE')

        frame = self.draw_detections(frame, detections, best)
        cv2.imshow('Sign Detection', frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    try:
        node = SignDetectionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
