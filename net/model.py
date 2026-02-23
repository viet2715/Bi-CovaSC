''' Torch '''
from net.attention import BAM_Block
from net.Covariance import SpatialCovarianceBlock, ChannelCovarianceBlock
from net.MAFE import MAFE_Block
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


class BiCovaSC(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=13):
        super(BiCovaSC, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.features = MAFE_Block(features=64)
        self.w = nn.Linear(10, 10)
        self.SpatialCov = SpatialCovarianceBlock()
        self.ChannelCov = ChannelCovarianceBlock()
        self.S_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Conv1d(1, 1, kernel_size=256, stride=256, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        )
        self.C_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Conv1d(1, 1, kernel_size=64, stride=64, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        )
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        self.pca = BAM_Block(64)
        self.psa = BAM_Block(64)

    def forward(self, input1, input2):
        q = self.features(input1)
        q_channel = self.pca(q)
        q_spatial = self.psa(q)
        B, C, H, W = q.size()
        S_channel = []
        S_spatial = []
        for i in range(len(input2)):
            s_safe = self.features(input2[i])
            S_channel.append(self.pca(s_safe))
            S_spatial.append(self.psa(s_safe))

        m_s = self.SpatialCov(q, S_spatial)
        m_s = self.S_classifier(m_s.view(m_s.size(0), 1, -1))
        m_s = m_s.squeeze(1)

        m_c = self.ChannelCov(q, S_channel)
        m_c = self.C_classifier(m_c.view(m_c.size(0), 1, -1))
        m_c = m_c.squeeze(1)

        total = self.w1 * m_s + self.w2 * m_c

        return total