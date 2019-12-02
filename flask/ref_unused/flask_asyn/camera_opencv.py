import sys
sys.path.append('../')
import os
import cv2
from base_camera import BaseCamera
import numpy as np


class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
            print('set_video_source:', int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        # camera.set(3,320) # width
        # camera.set(4,240) # hight
        print('width:',camera.get(3),'height',camera.get(4))
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()
            # origin:(480, 640, 3) uint8 # height width
            # print(img.shape, img.dtype) # <class 'numpy.ndarray'>
            yield img
