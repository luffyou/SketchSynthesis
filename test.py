import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from models import G, D, ResnetGenerator, weightsInit
from utils import isImageFile, loadImage, saveImage

import functools
import torch.nn as nn

from time import time

### Training parament setting
parser = argparse.ArgumentParser(description = 'implementation')
parser.add_argument('--cuda', action = 'store_true', help = 'use cuda?')
parser.add_argument('--img_name', type=str, required=True, help='img file to generate')
opt = parser.parse_args()
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

image_name = './static/upload/' + opt.img_name
img = loadImage(image_name, -1, -1, -1, 'Testing')
input = Variable(img).view(1, -1, image_sz, image_sz)
if opt.cuda:
    # netaG = netaG.cuda()
    photo_G_1 = photo_G_1.cuda()
    sketch_G_2 = sketch_G_2.cuda()
    input = input.cuda()

wp = photo_G_1(input)
out = sketch_G_2(wp)

out = out.cpu()
out_img = out.data[0]

if not os.path.exists("result"):
    os.mkdir("result")
saveImage(out_img, "result/{}".format(opt.img_name))

