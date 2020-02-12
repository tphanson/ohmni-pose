import numpy as np
import cv2 as cv
from PIL import ImageDraw

from utils import image
from pose_engine import PoseEngine
from utils.camera import Camera

LABEL_FILTER = []


def draw_text(draw, text):
    draw.text((10, 10), text, fill="red")


def draw_pose(draw, x, y, label, score):
    draw.ellipse([(x, y), (x+5, y+5)], fill="red")
    draw.text((x + 10, y + 10), '%s: %.2f' % (label, score), fill="red")


if __name__ == "__main__":
    engine = PoseEngine(
        'models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')
    cam = Camera()
    stream = cam.get_stream()

    print("You can press Q button to stop the test!")
    while True:
        timer = cv.getTickCount()

        print("===========================")
        img = stream.get()
        cv_img = cv.resize(img, (641, 481))
        pil_img = image.convert_cv_to_pil(cv_img)
        poses, inference_time = engine.DetectPosesInImage(cv_img)

        print('Inference time: {:.4f}'.format(inference_time/1000))
        drawed_img = ImageDraw.Draw(pil_img)
        for pose in poses:
            if pose.score < 0.4:
                continue
            draw_text(drawed_img, str(pose.score))
            for label, keypoint in pose.keypoints.items():
                x = keypoint.yx[1]
                y = keypoint.yx[0]
                score = keypoint.score
                draw_pose(drawed_img, x, y, label, score)

        # Calculate frames per second (FPS)
        print("Total Estimated Time: {:.4f}".format(
            (cv.getTickCount()-timer)/cv.getTickFrequency()))
        fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
        print("FPS: {:.1f}".format(fps))
        print("\n")

        cv.imshow("Video", image.convert_pil_to_cv(pil_img))
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cv.destroyWindow("Video")
    cam.terminate()
