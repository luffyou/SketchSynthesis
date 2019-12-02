import os
import sys
sys.path.append('../')
from importlib import import_module
from flask import Flask, render_template, Response
from flask import request, json, jsonify
import base64
import io

import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from models import G, D, ResnetGenerator, weightsInit
from utils import isImageFile, loadImage, saveImage, deProcessImg, preProcessImg, TimePoint
import functools
import torch.nn as nn
from time import time, strftime
import numpy as np
import cv2
from scipy.misc import imresize, imsave
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 设置gpu

class Sketch(object):
    response_img = b''
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


app = Flask(__name__) 
sketch = Sketch()
tp = TimePoint()

@app.route('/') 
def index():
    return render_template('index.html')


#设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'bmp'])
# save the image as a picture
@app.route('/img_upload', methods=['POST'])
def img_upload():
    tp.start()
    req_file = request.files['file']  # get the image
    print(type(req_file),req_file) # <class 'werkzeug.datastructures.FileStorage'> <FileStorage: 'blob' ('image/jpeg')>
    name_sub = req_file.filename.split('.')[1].lower()
    if name_sub not in ALLOWED_EXTENSIONS:
        res_str = "request image error"
        print(res_str)
        return res_str 

    img_name = 'static.{}'.format(name_sub)
    img_res_name = 'static_res.jpg'
    req_file.save(img_name)

    img = loadImage(img_name, -1, -1, -1, 'Testing')
    img_in = Variable(img).view(1, -1, sketch.IMG_SIZE_IN, sketch.IMG_SIZE_IN)
    if sketch.use_cuda:
        img_in = img_in.cuda()
    tp.getCost('pre')
    wp = sketch.photo_G_1(img_in)
    out = sketch.sketch_G_2(wp)
    tp.getCost('net')
    img_out = sketch.postProcess(out) # <class 'numpy.ndarray'> (480, 640, 3)
    img_resp = cv2.imencode('.jpg', img_out)[1].tostring() # <class 'bytes'>
    img_resp = base64.b64encode(img_resp) # <class 'bytes'>
    img_resp = "data:image/jpeg;base64,{}".format(img_resp.decode()) # <class 'str'
    tp.getCost('post')
    tp.getTotalCost()

    return img_resp

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000 , threaded=True)