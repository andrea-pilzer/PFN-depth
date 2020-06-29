# Code for
# Progressive Fusion for Unsupervised Binocular Depth Estimation using Cycled Networks
# Andrea Pilzer, Stéphane lathuilière, Dan Xu, Mihai Puscas, Elisa Ricci, Nicu Sebe
#
# TPAMI 2019, SI/RGBD Vision
#
# parts of the code from https://github.com/mrharicot/monodepth
#

from __future__ import division
import tensorflow as tf
from ops import *
from utils import *

def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)
