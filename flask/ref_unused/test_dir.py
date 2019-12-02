import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from models import G, D, ResnetGenerator, weightsInit
from utils import isImageFile, loadImage, saveImage, TimePoint

import functools
import torch.nn as nn

from time import time


os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 设置gpu

### Training parament setting
parser = argparse.ArgumentParser(description = 'implementation')
parser.add_argument('--cuda', type=bool ,default = True, help = 'use cuda?')
# parser.add_argument('--img_name', type=str, required=True, help='img file to generate')
parser.add_argument('--img_dir', type=str, default='static/imgs', help='img file to generate')
opt = parser.parse_args()
white_list = ['jpg', 'jpeg', 'png', 'bmp']
# print(opt)
### RGB chanels
chanels = 3
### image size
image_sz = 256

### cuda setting
if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

### uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms
cudnn.benchmark = True

pmodel_dir = "model/photo_G_1_model.pth"
photo_G_1_state_dict = torch.load(pmodel_dir)
photo_G_1 = ResnetGenerator(chanels, chanels, 64, norm_layer=functools.partial(nn.BatchNorm2d, affine=True), use_dropout=True)
photo_G_1.load_state_dict(photo_G_1_state_dict)

smodel_dir = "model/sketch_G_2_model.pth"
sketch_G_2_state_dict = torch.load(smodel_dir)
sketch_G_2 = G(chanels, chanels, 64)
sketch_G_2.load_state_dict(sketch_G_2_state_dict)

if opt.cuda:
    # netaG = netaG.cuda()
    photo_G_1 = photo_G_1.cuda()
    sketch_G_2 = sketch_G_2.cuda()
if not os.path.exists("result"):
    os.mkdir("result")

tp = TimePoint()
img_list = os.listdir(opt.img_dir)
for img_path in img_list:
    if img_path.rstrip().split('.')[-1].lower() not in white_list:
        continue
    image_name = os.path.join(opt.img_dir, img_path)
    tp.start()
    img = loadImage(image_name, -1, -1, -1, 'Testing')
    input = Variable(img).view(1, -1, image_sz, image_sz)
    if opt.cuda:
        input = input.cuda()
    tp.getCost('pre')
    wp = photo_G_1(input)
    out = sketch_G_2(wp)
    tp.getCost('net')
    out = out.cpu()
    out_img = out.data[0]
    saveImage(out_img, "result/{}".format(img_path))
    tp.getCost('post')
    tp.getTotalCost()

print('Finish')

