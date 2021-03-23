import os
import argparse
import numpy as np

os.system('rm -r output')

parser = argparse.ArgumentParser()
parser.add_argument('--test_dir', default='test', help='path to test folder')
parser.add_argument('--g_ckpt', default = '1.pt', help = 'path to generator checkpoint')

config = parser.parse_args()

os.mkdir('output')
for i,img in enumerate(os.listdir(config.test_dir)):
  os.system('python test_image.py --model_name {} --image_name {}'.format(
    config.g_ckpt,
    config.test_dir + '/' + img
  )) 
  print('Saving {}th image'.format(i))