import os
import sys
sys.path.append('../')
from importlib import import_module
from flask import Flask, render_template, Response

import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from models import G, D, ResnetGenerator, weightsInit
from utils import isImageFile, loadImage, saveImage, deProcessImg, preProcessImg, TimePoint
import functools
import torch.nn as nn
from time import time
import numpy as np
import cv2
from scipy.misc import imresize, imsave

os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 设置gpu

class Sketch(object):
    use_cuda = True
    cudnn.benchmark = True
    need_trans = True

    IMG_SIZE_IN = 256
    IMG_H = 480 # 250
    IMG_W = 640 # 200
    chanels = 3

    pmodel_dir = "../model/photo_G_1_model.pth"
    smodel_dir = "../model/sketch_G_2_model.pth"
    photo_G_1 = None
    sketch_G_2 = None
    # tp = TimePoint()
    def __init__(self):

        photo_G_1_state_dict = torch.load(self.pmodel_dir)
        self.photo_G_1 = ResnetGenerator(self.chanels, self.chanels, 64, norm_layer=functools.partial(nn.BatchNorm2d, affine=True), use_dropout=True)
        self.photo_G_1.load_state_dict(photo_G_1_state_dict)

        sketch_G_2_state_dict = torch.load(self.smodel_dir)
        self.sketch_G_2 = G(self.chanels, self.chanels, 64)
        self.sketch_G_2.load_state_dict(sketch_G_2_state_dict)

        if self.use_cuda:
            # netaG = netaG.cuda()
            self.photo_G_1 = self.photo_G_1.cuda()
            self.sketch_G_2 = self.sketch_G_2.cuda()
        print('Sketch init')

    def preProcess(self, img):
        # print(img.shape, type(img)) # (480, 640, 3) <class 'numpy.ndarray'>
        # if len(img.shape) < 3: # <class 'numpy.ndarray'>
        #     img = np.expand_dims(img, axis = 2)
        #     img = np.repeat(img, 3, axis = 2)
        # imsave('test_pre.jpg',img)
        # self.tp.start()
        img = imresize(img, (self.IMG_SIZE_IN, self.IMG_SIZE_IN))
        img = np.transpose(img, (2, 0, 1))
        # numpy.ndarray to FloatTensor
        img = torch.from_numpy(img)
        # self.tp.getCost('pre_resize')
        if self.use_cuda:
            img = img.cuda()
        # self.tp.getCost('pre_cuda')
        img = preProcessImg(img, self.use_cuda)
        # self.tp.getCost('pre_process') # major cost
        img = Variable(img).view(1, -1, self.IMG_SIZE_IN, self.IMG_SIZE_IN)
        return img

    def postProcess(self, img_gpu):
        # self.tp.start()
        img_gpu = deProcessImg(img_gpu.data[0], self.use_cuda)
        # self.tp.getCost('post_process') # major cost
        img = img_gpu.cpu() # <class 'torch.Tensor'>
        # self.tp.getCost('post_cpu')
        img = img.numpy()
        img *= 255.0
        img = img.clip(0, 255)
        img = np.transpose(img, (1, 2, 0))
        img = imresize(img, (self.IMG_H, self.IMG_W, 3)) # row col
        img = img.astype(np.uint8)
        # self.tp.getCost('post_numpy')
        # imsave('test_post.jpg', img)
        # print(img.shape, type(img)) # (250, 200, 3) <class 'numpy.ndarray'> 
        return img


class VideoCamera(object):
    def __init__(self):
        self.sketch = Sketch()
        self.video_source = 0
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            self.video_source = int(os.environ['OPENCV_CAMERA_SOURCE'])
        print('vidoe_source:', self.video_source)
        self.cap = cv2.VideoCapture(self.video_source) 
        # self.cap.set(3, 320) # width
        # self.cap.set(4, 240) # height; set camera
        print('width:',self.cap.get(3),'height',self.cap.get(4))
        print('VideoCamera init')
    
    def __del__(self):
        self.cap.release()
    
    def get_frame(self):
        success, img_np = self.cap.read()
        # ret, jpg = cv2.imencode('.jpg', img_np)
        # return jpg.tobytes()
        return img_np
 
app = Flask(__name__)
 
@app.route('/') 
def index():
    return render_template('index.html')
 
def gen(cam):
    # tp = TimePoint()
    while True:
        img = cam.get_frame() # 此方式对前端的压力很大
        # time_start = time()
        # tp.start()
        in_img = cam.sketch.preProcess(img)
        # img = imresize(img, (cam.sketch.IMG_H, cam.sketch.IMG_W, 3)) 
        # tp.getCost('pre')
        wp = cam.sketch.photo_G_1(in_img)
        out = cam.sketch.sketch_G_2(wp)
        # tp.getCost('net')
        out_img = cam.sketch.postProcess(out)
        out_img = np.hstack((img, out_img))
        # print(out_img.shape, out_img.dtype)
        out_img = cv2.imencode('.jpg', out_img)[1].tobytes()
        # tp.getCost('post')
        # time_end = time()
        # print('totall cost', time_end - time_start)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + out_img + b'\r\n\r\n')
 
@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000 , threaded=True)