''' Torch '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

''' Mamba '''
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from functools import partial
from typing import Optional, Callable
from timm.models.layers import DropPath
import math

''' Scikit '''
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

''' Scipy '''
import scipy.io
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat

''' Einops '''
from einops import rearrange, repeat
from einops import reduce

''' Other '''
from random import randint
import time
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import pandas as pd
import os,re
import errno
import random
import urllib.request as urllib
import librosa
import numpy as np
import cv2
import functools
from tqdm import tqdm
import seaborn as sns

import torch
import torch.nn as nn

class Spatial_Attention(nn.Module):
    def __init__(self, dim, reduction=4):
        super(Spatial_Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim // reduction),
            nn.Conv2d(dim // reduction, dim // reduction, kernel_size=3, stride=1, padding=1, groups=dim // reduction),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim // reduction),
            nn.Conv2d(dim // reduction, dim // reduction, kernel_size=3, stride=1, padding=1, groups=dim // reduction),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim // reduction),
            nn.Conv2d(dim // reduction, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        attention_map = self.attention(x)
        return attention_map 

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.dw = nn.Conv2d(channel, channel, kernel_size=9, groups=int(channel) if isinstance(channel, torch.Tensor) else channel, padding=4)

    def forward(self, x):
        batch_size, channel, _, _ = x.size()
        squeeze = F.adaptive_avg_pool2d(x, (1, 1))  
        squeeze = squeeze.view(batch_size, channel)  

        excitation = self.fc1(squeeze)
        excitation = F.relu(excitation)
        excitation = self.fc2(excitation)

        excitation = excitation.view(batch_size, channel, 1, 1)
        return  excitation
    
class BAM_Block(nn.Module):
  def __init__(self, dim):
    super(BAM_Block, self).__init__()
    self.se = SEBlock(dim)
    self.spatial = Spatial_Attention(dim)
    self.sigmoid = nn.Sigmoid()
  def forward(self,x):
    attention = self.se(x) + self.spatial(x)
    attention = self.sigmoid(attention)
    return x*attention + x