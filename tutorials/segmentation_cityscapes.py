# we already use multiprocessing, multithreading will not improve anything here so disable it
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
import torchvision
import numpy as np
import trw



dataset = trw.datasets.create_cityscapes_dataset()

print('DONE')