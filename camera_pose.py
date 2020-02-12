import numpy as np
import cv2 as cv
from PIL import ImageDraw

from utils import image
from pose_engine import PoseEngine
from utils.camera import Camera

# 'nose','left eye','right eye','left ear','right ear',
# 'left shoulder','right shoulder','left elbow','right elbow','left wrist','right wrist',
# 'left hip','right hip','left knee','right knee','left ankle','right ankle'
LABEL_FILTER = ['left elbow', 'right elbow', 'left wrist', 'right wrist']


class Pose():
    def __init__(self):
        self.engine = PoseEngine(
            'models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')
        self.confidence = 3
        self.marin = 50
        self.image_shape = (640, 480)

    def draw_text(self, draw, text):
        draw.text((10, 10), text, fill="red")

    def draw_pose(self, draw, x, y, label, score):
        draw.ellipse([(x, y), (x+5, y+5)], fill="red")
        draw.text((x + 10, y + 10), '%s: %.2f' % (label, score), fill="red")

    def generate_bbox(self, marks):
        xmin, xmax, ymin, ymax = 10000, 0, 10000, 0
        for (_, x, y) in marks:
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
        for (label, x, y) in marks:
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
        for (label, x, y) in marks:
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
        bbox = [0, 0, 0, 0]
        if activated:
            bbox = self.generate_bbox(marks)
        return activated, bbox

    def predict(self):
        cam = Camera()
        stream = cam.get_stream()

        while True:
            timer = cv.getTickCount()

            print("===========================")
            img = stream.get()
            cv_img = cv.resize(img, (641, 481))
            pil_img = image.convert_cv_to_pil(cv_img)
            poses, inference_time = self.engine.DetectPosesInImage(cv_img)
            obj = None

            print('Inference time: {:.4f}'.format(inference_time/1000))
            drawed_img = ImageDraw.Draw(pil_img)
            for pose in poses:
                if pose.score < 0.4:
                    continue
                marks = []
                for label, keypoint in pose.keypoints.items():
                    x = keypoint.yx[1]
                    y = keypoint.yx[0]
                    score = keypoint.score
                    if label in LABEL_FILTER:
                        self.draw_pose(drawed_img, x, y, label, score)
                    marks.append((label, x, y))
                # Find an activation
                activated, bbox = self.activate(marks)
                (xmin, xmax, ymin, ymax) = bbox
                if activated:
                    self.draw_text(drawed_img, 'Activated')
                    obj = cv_img[ymin:ymax, xmin:xmax]
                else:
                    self.draw_text(drawed_img, 'Idle')

            # Calculate frames per second (FPS)
            print('Total Estimated Time: {:.4f}'.format(
                (cv.getTickCount()-timer)/cv.getTickFrequency()))
            fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
            print('FPS: {:.1f}'.format(fps))
            print('\n')

            if obj is not None:
                obj = cv.resize(obj, (96, 96))
                cv.imshow('Activation', obj)
                cv.moveWindow('Activation', 90, 650)
            cv.imshow('Video', image.convert_pil_to_cv(pil_img))
            if cv.waitKey(10) & 0xFF == ord('q'):
                break

        cv.destroyWindow('Activation')
        cv.destroyWindow('Video')
        cam.terminate()


if __name__ == "__main__":
    pose = Pose()
    pose.predict()
