import numpy as np
import cv2 as cv

from pose_engine import PoseEngine

# 'nose','left eye','right eye','left ear','right ear',
# 'left shoulder','right shoulder','left elbow','right elbow','left wrist','right wrist',
# 'left hip','right hip','left knee','right knee','left ankle','right ankle'


class Pose():
    def __init__(self):
        self.engine = PoseEngine(
            'models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')
        self.confidence = 3
        self.marin = 50
        self.image_shape = (640, 480)
        self.input_shape = (641, 481)

    def generate_bbox(self, marks):
        xmin, xmax, ymin, ymax = 10000, 0, 10000, 0
        for (_, _, x, y) in marks:
            if x <= xmin:
                xmin = int(x)
            if x >= xmax:
                xmax = int(x)
            if y <= ymin:
                ymin = int(y)
            if y >= ymax:
                ymax = int(y)
        xmin -= self.marin
        xmax += self.marin
        ymin -= self.marin
        ymax += self.marin
        if xmin < 0:
            xmin = 0
        if xmax > self.image_shape[0]:
            xmax = self.image_shape[0]
        if ymin < 0:
            ymin = 0
        if ymax > self.image_shape[1]:
            ymax = self.image_shape[1]
        return (xmin, xmax, ymin, ymax)

    def activate_by_left_hand(self, marks):
        dx, dy = 0, 0
        for (label, _, x, y) in marks:
            if label == 'left elbow':
                dx += x
                dy += y
            if label == 'left wrist':
                dx -= x
                dy -= y
        if dy > self.confidence * np.abs(dx):
            return True
        else:
            return False

    def activate_by_right_hand(self, marks):
        dx = 0
        dy = 0
        for (label, _, x, y) in marks:
            if label == 'right elbow':
                dx += x
                dy += y
            if label == 'right wrist':
                dx -= x
                dy -= y
        if dy > self.confidence * np.abs(dx):
            return True
        else:
            return False

    def activate(self, marks):
        activated = False
        activated = self.activate_by_left_hand(
            marks) or self.activate_by_right_hand(marks)
        bbox = (0, 0, 0, 0)
        if activated:
            bbox = self.generate_bbox(marks)
        return activated, bbox

    def inference(self, img):
        if img.shape[0] != self.input_shape[1] or img.shape[1] != self.input_shape[0]:
            raise ValueError(
                'Input shape is not correct. Refer pose.input_shape for details.')

        poses, inference_time = self.engine.DetectPosesInImage(img)
        objects = []
        for pose in poses:
            if pose.score < 0.4:
                continue
            marks = []
            for label, keypoint in pose.keypoints.items():
                x = keypoint.yx[1]
                y = keypoint.yx[0]
                score = keypoint.score
                marks.append((label, score, x, y))
            objects.append(marks)
        return objects, inference_time

    def predict(self, img):
        objects, inference_time = self.inference(img)

        activated = False
        obj_img = None
        bbox = None
        for marks in objects:
            # Find an activation
            activated, bbox = self.activate(marks)
            (xmin, xmax, ymin, ymax) = bbox
            if activated:
                obj_img = img[ymin:ymax, xmin:xmax]
                obj_img = cv.resize(obj_img, (96, 96))
        return activated, obj_img, bbox, objects, inference_time
