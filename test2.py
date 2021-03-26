import argparse
import time
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
import os
from model import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--test_dir', default = '/content/SRGAN/data/celeba_hq/val/male' ,type=str, help='test directory')
parser.add_argument('--model_name', default='netG_epoch_4_1.pth', type=str, help='generator model epoch name')
parser.add_argument('--img_size', default=256, type=int, help='load img size')
opt = parser.parse_args()

os.system('rm -r output')

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False

MODEL_NAME = opt.model_name

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))

os.mkdir('output')
for i,img in enumerate(os.listdir(opt.test_dir)):

  image = Image.open(opt.test_dir + '/' + img)
  image = image.resize((opt.img_size, opt.img_size))
  image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)

  if TEST_MODE:
    image = image.cuda()

  start = time.clock()
  out = model(image)
  elapsed = (time.clock() - start)
  print('cost' + str(elapsed) + 's')
  out_img = ToPILImage()(out[0].data.cpu())
  out_img.save('output/' + img) 
  print('Saving {}th image'.format(i))



