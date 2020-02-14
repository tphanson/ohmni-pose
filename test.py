import cv2 as cv
from PIL import ImageDraw

from utils import image
from pose import Pose
from utils.camera import Camera

# 'nose','left eye','right eye','left ear','right ear',
# 'left shoulder','right shoulder','left elbow','right elbow','left wrist','right wrist',
# 'left hip','right hip','left knee','right knee','left ankle','right ankle'
LABEL_FILTER = ['left elbow', 'right elbow', 'left wrist', 'right wrist']


class Test():
    def __init__(self):
        self.pose = Pose()

    def draw_text(self, draw, text):
        draw.text((10, 10), text, fill="red")

    def draw_pose(self, draw, x, y, label, score):
        draw.ellipse([(x, y), (x+5, y+5)], fill="red")
        draw.text((x + 10, y + 10), '%s: %.2f' % (label, score), fill="red")

    def test(self):
        cam = Camera()
        stream = cam.get_stream()

        while True:
            timer = cv.getTickCount()

            print("===========================")
            img = stream.get()
            cv_img = cv.resize(img, self.pose.input_shape)
            pil_img = image.convert_cv_to_pil(cv_img)
            objects, time, status, obj_img, bbox = self.pose.predict(cv_img)

            print('Inference time: {:.4f}'.format(time/1000))
            drawed_img = ImageDraw.Draw(pil_img)
            for marks in objects:
                for mark in marks:
                    (label, score, x, y) = mark
                    if label in LABEL_FILTER:
                        self.draw_pose(drawed_img, x, y, label, score)

            # Calculate frames per second (FPS)
            print('Total Estimated Time: {:.4f}'.format(
                (cv.getTickCount()-timer)/cv.getTickFrequency()))
            fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
            print('FPS: {:.1f}'.format(fps))
            print('\n')

            if status != 0:
                self.draw_text(drawed_img, 'Activated - Status: '+str(status))
                cv.imshow('Activation', obj_img)
                cv.moveWindow('Activation', 90, 650)
            else:
                self.draw_text(drawed_img, 'Idle')
            cv.imshow('Video', image.convert_pil_to_cv(pil_img))
            if cv.waitKey(10) & 0xFF == ord('q'):
                break

        cv.destroyWindow('Activation')
        cv.destroyWindow('Video')
        cam.terminate()


if __name__ == "__main__":
    test = Test()
    test.test()
