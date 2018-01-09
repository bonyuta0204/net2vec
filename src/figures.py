import glob
import os
import sys
import numpy as np
import re
#import upsample
import time
import loadseg
from scipy.misc import imread, imresize, imsave
from loadseg import normalize_label
import expdir
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

from linearprobe_pytorch import CustomLayer

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(int(100 * y))

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


def get_single_acts(acts, thresh, filter_i, up_size=(224,224), thresh_first=True):
    upsample = nn.Upsample(size=up_size, mode='bilinear')
    if thresh_first:
        return upsample(Variable(torch.Tensor((acts > thresh).astype(float)))).data.cpu().numpy()[0,filter_i]
    else:
        return (upsample(Variable(torch.Tensor(acts))).data.cpu().numpy()[0] > thresh).astype(float)[filter_i]


def get_weighted_acts(acts, thresh, label_weights, label_bias=None,up_size=(224,224), act=True, 
                      positive=False, bias=False, thresh_first=True):
    layer = CustomLayer(len(label_weights), upsample=thresh_first, up_size=up_size, act=act, positive=positive, bias=bias)
    layer.weight.data[...] = torch.Tensor(label_weights)
    if bias:
        assert(label_bias is not None)
        layer.bias.data[...] = torch.Tensor(label_bias)
    if thresh_first:
        return layer(Variable(torch.Tensor((acts > thresh).astype(float)))).data.cpu().numpy()
    else:
        upsample = nn.Upsample(size=up_size, mode='bilinear')
        return layer((upsample(Variable(torch.Tensor(acts))) > Variable(torch.Tensor(thresh))).type(torch.FloatTensor)).data.cpu().numpy()


def show_examples_for_unit(ds, blobdata, thresh, blob, unit_i, examples_idx, up_size=(224,224), num_to_show=8, mask_alpha=0.2, thresh_first=False):
    f, ax = plt.subplots(1,num_to_show,figsize=(4*num_to_show,4))
    for i in range(num_to_show):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        if i == 0:
            pass
            #ax[i].set_ylabel('unit %d' % unit_i)
        #try:
        up = get_single_acts(np.expand_dims(blobdata[blob][examples_idx[i]], axis=0), thresh[blob], unit_i, up_size=up_size, thresh_first=thresh_first)
        ax[i].imshow((np.expand_dims((up > 0.5), axis=2) * (1-mask_alpha) + mask_alpha) * imread(ds.filename(examples_idx[i]))/255.)
        #    #ax[i].imshow(probe.activation_visualization(blob, unit_i, top[unit_i][i], normalize=True, use_fieldmap=False))
        #except:
        #    break
    f.subplots_adjust(wspace=0.0,hspace=0.0)
    #plt.tight_layout()
    plt.show()


def show_examples_for_linear(ds, blobdata, thresh, blob, label_weights, examples_idx, up_size=(224,224), num_to_show=8, mask_alpha=0.2, 
                             thresh_first=False):
    f, ax = plt.subplots(1,num_to_show,figsize=(4*num_to_show,4))
    for i in range(num_to_show):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        if i == 0:
            pass
            #ax[i].set_ylabel('unit %d' % unit_i)
        #try:
        up = get_weighted_acts(np.expand_dims(blobdata[blob][examples_idx[i]], axis=0), thresh[blob], label_weights, label_bias=None,
                          up_size=up_size, act=True, positive=False, bias=False, thresh_first=thresh_first)
        #except:
        #    break
        ax[i].imshow((np.expand_dims((up > 0.5), axis=2) * (1-mask_alpha) + mask_alpha) * imread(ds.filename(examples_idx[i]))/255.)

    f.subplots_adjust(wspace=0.0,hspace=0.0)
    plt.show()

def find_label(phrase, label_names):
    return np.array([i for i in range(len(label_names)) if phrase in label_names[i]])

def find_exact_label(phrase, label_names):
    for i in range(len(label_names)):
        if phrase == label_names[i]:
            return i
    else:
        return -1
