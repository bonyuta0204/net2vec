import os, sys, time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import pylab
import matplotlib.pyplot as plt

from scipy.misc import imread, imresize
from scipy.stats import pearsonr

from invert import invert

from indexdata import load_image_to_label
from indexdataset import load_image_dataset_label_index
from voclabels import voc_labels

%matplotlib inline

%load_ext autoreload
%autoreload 2

import loadseg
import expdir
import intersect
import upsample
from labelprobe import cached_memmap

import sys
from viewprobe import NetworkProbe

import matplotlib
from matplotlib.ticker import FuncFormatter

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(int(100 * y))

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'

import matplotlib.pylab as pylab
params = {'axes.labelsize': 16,
         'axes.titlesize':16,
         'xtick.labelsize':14,
         'ytick.labelsize':14,
         'legend.fontsize':14}
pylab.rcParams.update(params)

gpu = 1

cuda = True if gpu is not None else False
use_mult_gpu = isinstance(gpu, list)
if cuda:
    if use_mult_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu).strip('[').strip(']')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpu
print(torch.cuda.device_count(), use_mult_gpu, cuda)

blob_names = {
    'features.1': 'relu1',
    'features.4': 'relu2',
    'features.7': 'relu3',
    'features.9': 'relu4',
    'features.11': 'relu5',
    'features': 'pool5'
}

directory='/home/ruthfong/NetDissect/probes/pytorch_alexnet_imagenet'
if not os.path.exists(directory):
    directory='/scratch/shared/slow/ruthfong/pytorch_alexnet_imagenet'
    assert(os.path.exists(directory))
blob='features.11'

ed = expdir.ExperimentDirectory(directory)

info = ed.load_info()
blob_info = ed.load_info(blob=blob)
shape = blob_info.shape
ds = loadseg.SegmentationData(info.dataset)
categories = ds.category_names()

K = shape[1]
L = ds.label_size()
N = ds.size()

label_names = np.array([ds.label[i]['name'] for i in range(L)])

blobs = ['features.1', 'features.4', 'features.7', 'features.9', 'features.11']

suffix=''

linear_ind_ious = {}
single_ind_ious = {}
linear_set_ious = {}
weights = {}

single_set_train_ious = {}
single_set_val_ious = {}

for blob in blobs:
    blob_info = ed.load_info(blob=blob)
    shape = blob_info.shape
    K = shape[1]
    linear_ind_ious[blob] = ed.open_mmap(blob=blob, part='linear_ind_ious', mode='r', dtype='float32', shape=(L,N))
    single_ind_ious[blob] = ed.open_mmap(blob=blob, part='single_ind_ious', mode='r', dtype='float32', suffix=suffix, shape=(L,N,K))
    linear_set_ious[blob] = ed.open_mmap(blob=blob, part='linear_set_val_ious', mode='r', dtype='float32', shape=(L,))
    single_set_train_ious[blob] = ed.open_mmap(blob=blob, part='single_set_train_ious', mode='r', dtype='float32', suffix=suffix, shape=(L,K))
    single_set_val_ious[blob] = ed.open_mmap(blob=blob, part='single_set_val_ious', mode='r', dtype='float32', suffix=suffix, shape=(L,K))
    weights[blob] = ed.open_mmap(blob=blob, part='linear_weights', mode='r', dtype='float32', shape=(L,K))
    
linear_ind_ious_val = {}
linear_set_ious_val = {}
linear_set_train_ious_val = {}
linear_set_val_ious_val = {}
weights_val = {}

suffix = '_validation'
for blob in ['features.9', 'features.11']:
    blob_info = ed.load_info(blob=blob)
    shape = blob_info.shape
    K = shape[1]
    linear_ind_ious_val[blob] = ed.open_mmap(blob=blob, part='linear_ind_ious%s' % suffix, mode='r', dtype='float32', shape=(L,N))
    linear_set_ious_val[blob] = ed.open_mmap(blob=blob, part='linear_set_ious%s' % suffix, mode='r', dtype='float32', shape=(L,))
    linear_set_train_ious_val[blob] = ed.open_mmap(blob=blob, part='linear_set_train_ious%s' % suffix, mode='r', dtype='float32', shape=(L,))
    linear_set_val_ious_val[blob] = ed.open_mmap(blob=blob, part='linear_set_val_ious%s' % suffix, mode='r', dtype='float32', shape=(L,))
    weights_val[blob] = ed.open_mmap(blob=blob, part='linear_weights%s' % suffix, mode='r', dtype='float32', shape=(L,K))

thresh = {}
quantile = 0.005
qcode = ('%f' % quantile).replace('0.','').rstrip('0')

for blob in blobs: 
    blob_info = ed.load_info(blob=blob)
    shape = blob_info.shape
    K = shape[1]
    
    quantdata = ed.open_mmap(blob=blob, part='quant-*', shape=(K, -1))
    threshold = quantdata[:, int(round(quantdata.shape[1] * quantile))]
    thresh[blob] = np.copy(threshold[:, np.newaxis, np.newaxis])

train_idx = np.array([ds.split(i) == 'train' for i in range(N)])
val_idx = np.array([ds.split(i) == 'val' for i in range(N)])
train_ind = np.array([True if ds.split(i) == 'train' else False for i in range(N)])
val_ind = np.array([True if ds.split(i) == 'val' else False for i in range(N)])

image_to_label = load_image_to_label(directory)

pc = ds.primary_categories_per_index()
pc[0] = -1
categories = np.array(ds.category_names())
print categories

suffix = '_lr_1e-1_sgd_quant_1_epochs_30_iter_15_num_filters_2'
#num_filters = [1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 80, 100, 128]
#F = len(num_filters) + 1
F = 2
epochs = 30

disc_weights = {}
disc_bias = {}
disc_results = {}

for blob in blobs:
    blob_info = ed.load_info(blob=blob)
    shape = blob_info.shape
    K = shape[1]
    disc_weights[blob] = ed.open_mmap(blob=blob, part='linear_weights_disc%s' % suffix, mode='r', dtype='float32', shape=(L, F, K))
    disc_bias[blob] = ed.open_mmap(blob=blob, part='linear_bias_disc%s' % suffix, mode='r', dtype='float32', shape=(L, F))
    disc_results[blob] = ed.open_mmap(blob=blob, part='linear_results_disc%s' % suffix, mode='r', dtype='float32', shape=(L, F, 4, epochs))

suffix = '_lr_1e-1_sgd_quant_1_epochs_30_iter_15_num_filters'
num_filters = [1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 80, 100, 128]
F = len(num_filters) + 1

disc_weights_all = {}
disc_bias_all = {}
disc_results_all = {}

for blob in blobs[2:]:
    blob_info = ed.load_info(blob=blob)
    shape = blob_info.shape
    K = shape[1]

    disc_weights_all[blob] = ed.open_mmap(blob=blob, part='linear_weights_disc%s' % suffix, mode='r', dtype='float32', shape=(L, F, K))
    disc_bias_all[blob] = ed.open_mmap(blob=blob, part='linear_bias_disc%s' % suffix, mode='r', dtype='float32', shape=(L, F))
    disc_results_all[blob] = ed.open_mmap(blob=blob, part='linear_results_disc%s' % suffix, mode='r', dtype='float32', shape=(L, F, 4, epochs))

#suffix = '_lr_1e-1_sgd_quant_1_epochs_30_iter_15'

#disc_weights_last = ed.open_mmap(blob=blob, part='linear_weights_disc%s' % suffix, mode='w+', dtype='float32', shape=(L,2,K))
#disc_bias_last = ed.open_mmap(blob=blob, part='linear_bias_disc%s' % suffix, mode='w+', dtype='float32', shape=(L,2))
#disc_results_last = ed.open_mmap(blob=blob, part='linear_results_disc%s' % suffix, mode='w+', dtype='float32', shape=(L,2,4,epochs))

#disc_weights_last[:,0,:] = disc_weights_all[:,0,:]
#disc_bias_last[:,0] = disc_bias_all[:,0]
#disc_results_last[:,0,:,:] = disc_results_all[:,0,:,:]

#disc_weights_last[:,1,:] = disc_weights_all[:,-1,:]
#disc_bias_last[:,1] = disc_bias_all[:,-1]
#disc_results_last[:,1,:,:] = disc_results_all[:,-1,:,:]

#ed.finish_mmap(disc_weights_last)
#ed.finish_mmap(disc_bias_last)
#ed.finish_mmap(disc_results_last)

label_i = 154

f, ax = plt.subplots(1,2, figsize=(12,4))
filters_range = np.concatenate((num_filters, [256]))
ax[0].plot(filters_range, disc_results_all[blob][label_i, :, :2, -1])
ax[1].plot(filters_range, disc_results_all[blob][label_i, :, 2:, -1])
first = disc_results[blob][label_i,0,-1,-1]
last = disc_results[blob][label_i,-1,-1,-1]
thresh = first + (last-first)*0.90
ax[1].plot(filters_range, len(filters_range) * [thresh])
(a, b) = np.polyfit(np.log(filters_range), disc_results_all[blob][label_i, :,-1, -1], 1)
ax[1].plot(filters_range, a*np.log(filters_range) + b)
#(a1, b1) = np.polyfit(filters_range, np.log(results[-1]), 1, w=np.sqrt(results[-1]))
#ax[1].plot(filters_range, np.exp(b1)*np.exp(a1*filters_range))
#ax[1].plot(filters_range, 0.10*np.log10(filters_range-1)+0.55)
ax[0].set_title('BCE Loss')
ax[1].set_title('One vs. Rest Accuracy (%d: %s, %s)' % (label_i, label_names[label_i], blob_names[blob]))
ax[0].set_xlabel('# Filters')
ax[1].set_xlabel('# Filters')
plt.legend(['train','val','90% thresh', '%.3f*log(x)+%.3f)' % (a, b)])
plt.show()
print b/a

f, ax = plt.subplots(1,1)
for blob in blobs[2:]:
    blob_info = ed.load_info(blob=blob)
    filters_range = np.concatenate((num_filters, [blob_info.shape[1]]))
    x = disc_results_all[blob][:, :, -1, -1]
    ax.errorbar(filters_range, np.mean(x[np.isfinite(x[:,-1])], axis=0), 
                yerr=np.std(x[np.isfinite(x[:,-1])], axis=0)/np.sqrt(x.shape[0]), label=blob_names[blob])
ax.set_title(r'Classification Accuracy for Different $F$')
ax.set_xlabel(r'Number of Filters Used ($F$)')
ax.set_ylabel('Mean Val Accuracy')
ax.legend()
#plt.ylim([0.5, 1.0])
plt.show()

for blob in blobs[2:]:
    f, ax = plt.subplots(1,1)
    blob_info = ed.load_info(blob=blob)
    filters_range = np.concatenate((num_filters, [blob_info.shape[1]]))
    x = disc_results_all[blob][:, :, -1, -1]
    for cat_i in range(len(categories)):
        y = [i for i in range(L) if pc[i] == cat_i]
        x = x[y]
    ax.errorbar(filters_range, np.mean(x[np.isfinite(x[:,-1])], axis=0), 
                yerr=np.std(x[np.isfinite(x[:,-1])], axis=0)/np.sqrt(x.shape[0]), label=blob_names[blob])
ax.set_title('Classification Accuracy Using Different # of Filters')
ax.set_xlabel('# Filters')
ax.set_ylabel('One vs. Rest Mean Val Accuracy')
ax.legend()
#plt.ylim([0.5, 1.0])
plt.show()

num_filters = [1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 80, 100, 128]

perc_thresh = 0.90
for perc_thresh in [0.5, 0.6, 0.7, 0.8,0.9]:
    f, ax = plt.subplots(1,1)
    for blob in blobs[2:]:
        blob_info = ed.load_info(blob=blob)
        shape = blob_info.shape
        K = shape[1]

        filters_range = np.concatenate((num_filters, [K]))

        crit_filters = np.zeros(L)
        for label_i in range(1,L):

            first = disc_results_all[blob][label_i,0,-1,-1]
            last = disc_results_all[blob][label_i,-1,-1,-1]
            #thresh = first + (last-first)*perc_thresh
            thresh = 0.5 + (last-0.5) * perc_thresh 
            try:
                crit_filters[label_i] = num_filters[np.where(disc_results_all[blob][label_i,:,-1,-1] >= thresh)[0][0]]
            except:
                crit_filters[label_i] = K
            crit_filters_count = []
            for f in filters_range:
                crit_filters_count.append(np.sum(crit_filters == f))
        ax.plot(np.true_divide(filters_range, K), np.cumsum(crit_filters_count)/float(np.sum(crit_filters_count)), '.-', label=blob_names[blob])
    ax.set_xlabel(r'$\leq$% of Filters Used ($\leq F$/$K$)')
    ax.set_ylabel(r'%% Concepts %d%% Enc w $\leq F$ Filters' % int(perc_thresh*100), fontsize=14)
    ax.set_title('%% Filters needed to Encode %d%% of Concept' % (int(perc_thresh*100)))
    ax.legend()
    #plt.xticks(range(len(num_filters) + 1), map(str, filters_range))

    # Create the formatter using the function to_percent. This multiplies all the
    # default labels by 100, making them all percentages
    formatter = FuncFormatter(to_percent)
    # Set the formatter
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()

num_filters = np.array([1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 80, 100, 128])

cat_idx = range(6)

perc_thresh = 0.90
for blob in blobs[2:]:
    f, ax = plt.subplots(1,1)
    blob_info = ed.load_info(blob=blob)
    shape = blob_info.shape
    K = shape[1]

    filters_range = np.concatenate((num_filters, [K]))

    crit_filters = np.zeros(L)
    for label_i in range(1,L):
        
        first = disc_results_all[blob][label_i,0,-1,-1]
        last = disc_results_all[blob][label_i,-1,-1,-1]
        #thresh = first + (last-first)*perc_thresh
        thresh = 0.5 + (last-0.5) * perc_thresh 
        try:
            crit_filters[label_i] = num_filters[np.where(disc_results_all[blob][label_i,:,-1,-1] >= thresh)[0][0]]
        except:
            crit_filters[label_i] = K
            
    crit_filters_count = np.zeros((len(cat_idx), len(filters_range)))
    for i in cat_idx:
        for j in range(len(filters_range)):
            f = filters_range[j]
            crit_filters_count[i][j] = np.sum(crit_filters[pc == i] == f)

    for i in range(len(cat_idx)):
        ax.plot(filters_range, np.cumsum(crit_filters_count[i,:])/float(np.sum(pc == i)), '.-', label=categories[i])
    ax.set_xlabel('Number of Filters')
    ax.set_ylabel(r'%% Concepts %d%% Enc w $\leq N$ Filters' % int(perc_thresh*100), fontsize=14)
    ax.set_title('# Filters to Encode %d%% Concept (%s)' % (int(perc_thresh*100), blob_names[blob]))
    ax.legend()

    formatter = FuncFormatter(to_percent)
    # Set the formatter
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.show()

f, ax = plt.subplots(1,2, figsize=(8,4))
ax[0].bar(range(len(num_filters) + 1), crit_filters_count)
ax[1].bar(range(len(num_filters) + 1), np.cumsum(crit_filters_count))
for a in ax:
    a.set_xlabel('Number of Filters')
    a.set_ylabel('Number of Concepts')
    a.set_title('Number of Filters needed to Encode %d%% Concept (%s)' % (int(perc_thresh*100), blob_names[blob]))
plt.xticks(range(len(num_filters) + 1), map(str, filters_range))
plt.show()

ed.has_mmap(blob=blob, part='linear_set_val_ious_num_filters_160', inc=True)

L = 1198
blob = 'features.11'
num_filters = np.array([1, 2, 4, 8, 16, 32, 64, 128,160,192,224])
suffixes = ['_num_filters_%d' % n for n in num_filters]
suffixes.append('')

linear_set_ious_num_filters = np.zeros((L,len(suffixes)))
for i in range(len(suffixes)):
    linear_set_ious_num_filters[:,i] = ed.open_mmap(blob=blob, part='linear_set_val_ious%s' % suffixes[i], mode='r', 
                                               dtype='float32', shape=(L,))

idx = np.array([i for i in range(len(pc)) if (pc[i] < 4 and pc[i] >= 0) and i >= 100])
f, ax = plt.subplots(1,1)
ax.errorbar(np.concatenate((num_filters, [256])), np.mean(linear_set_ious_num_filters[idx], axis=0), 
           yerr=np.std(linear_set_ious_num_filters[idx], axis=0)/np.sqrt(len(idx)))
plt.show()

label_i = 154
f, ax = plt.subplots(1,1)
ax.plot(np.concatenate((num_filters, [256])), linear_set_ious_num_filters[label_i])
ax.plot(np.concatenate((num_filters, [256])), linear_set_ious_num_filters[label_i])
plt.show()

idx = np.array([i for i in range(len(pc)) if (pc[i] < 4 and pc[i] >= 0) and i >= 100])

perc_thresh = 0.90
f, ax = plt.subplots(1,1)
blob_info = ed.load_info(blob=blob)
shape = blob_info.shape
K = shape[1]

filters_range = np.concatenate((num_filters, [K]))

crit_filters = np.zeros(L)
for label_i in idx:

    first = linear_set_ious_num_filters[label_i,0]
    last = linear_set_ious_num_filters[label_i,-1]
    #thresh = first + (last-first)*perc_thresh
    thresh = last * perc_thresh 
    try:
        crit_filters[label_i] = num_filters[np.where(linear_set_ious_num_filters[label_i,:] >= thresh)[0][0]]
    except:
        crit_filters[label_i] = K
    crit_filters_count = []
    for f in filters_range:
        crit_filters_count.append(np.sum(crit_filters == f))
ax.plot(np.true_divide(filters_range, K), np.cumsum(crit_filters_count)/float(np.sum(crit_filters_count)), '.-', label=blob_names[blob])
ax.set_xlabel(r'$\leq$% of Filters Used ($\leq F$/$K$)')
ax.set_ylabel(r'%% Concepts %d%% Enc w $\leq F$ Filters' % int(perc_thresh*100), fontsize=14)
ax.set_title('# Filters to Encode %d%% Concept' % (int(perc_thresh*100)))
ax.legend()
#plt.xticks(range(len(num_filters) + 1), map(str, filters_range))

# Create the formatter using the function to_percent. This multiplies all the
# default labels by 100, making them all percentages
formatter = FuncFormatter(to_percent)
# Set the formatter
plt.gca().yaxis.set_major_formatter(formatter)
plt.gca().xaxis.set_major_formatter(formatter)
plt.show()

cat_idx = range(1,4)

perc_thresh = 0.80
f, ax = plt.subplots(1,1)
blob_info = ed.load_info(blob=blob)
shape = blob_info.shape
K = shape[1]

filters_range = np.concatenate((num_filters, [K]))

crit_filters = np.zeros(L)
for label_i in idx:
    first = linear_set_ious_num_filters[label_i,0]
    last = linear_set_ious_num_filters[label_i,-1]
    #thresh = first + (last-first)*perc_thresh
    thresh = last * perc_thresh 
    try:
        crit_filters[label_i] = num_filters[np.where(linear_set_ious_num_filters[label_i,:] >= thresh)[0][0]]
    except:
        crit_filters[label_i] = K

        [x for x in firsts[pc == i] if not np.isnan(x)]) for i in cat_idx]
crit_filters_count = np.zeros((len(cat_idx), len(filters_range)))
for i in range(len(cat_idx)):
    for j in range(len(filters_range)):
        f = filters_range[j]
        crit_filters_count[i][j] = np.sum(crit_filters[pc == cat_idx[i]] == f)

for i in range(len(cat_idx)):
    ax.plot(filters_range, np.cumsum(crit_filters_count[i,:])/float(np.sum(pc == cat_idx[i])), '.-', label=categories[cat_idx[i]])
ax.set_xlabel('Number of Filters')
ax.set_ylabel(r'%% of Concepts Encoding %d%% With $\leq N$ Filters' % int(perc_thresh*100))
ax.set_title('Number of Filters needed to Encode %d%% Concept (%s)' % (int(perc_thresh*100), blob_names[blob]))
ax.legend()

formatter = FuncFormatter(to_percent)
# Set the formatter
plt.gca().yaxis.set_major_formatter(formatter)

plt.show()

for blob in blobs[2:]:
    firsts = disc_results[blob][:,0,-1,-1]
    lasts = disc_results[blob][:,-1,-1,-1]

    f, ax = plt.subplots(1,1)

    cat_idx = range(6) # exclude scene and texture

    idx = np.arange(len(cat_idx))
    width = 1/3.
    counts = np.array([np.sum(pc == i) for i in cat_idx])

    single_mus = np.array([np.mean([x for x in firsts[pc == i] if not np.isnan(x)]) for i in cat_idx])
    dist_mus = np.array([np.mean([x for x in lasts[pc == i] if not np.isnan(x)]) for i in cat_idx])
    single_sigmas = np.array([np.std([x for x in firsts[pc == i] if not np.isnan(x)]) for i in cat_idx])
    dist_sigmas = np.array([np.std([x for x in lasts[pc == i] if not np.isnan(x)]) for i in cat_idx])
    ax.bar(idx, single_mus, width, yerr=np.true_divide(single_sigmas, np.sqrt(counts)), label='single')
    ax.bar(idx + width, dist_mus, width, yerr=np.true_divide(dist_sigmas, np.sqrt(counts)), label='dist')

    #ax.bar(idx+width, mus_single, width, yerr=np.true_divide(sigmas_single, np.sqrt(counts)), label='best filter')
    ax.legend()
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('One vs. Rest Classification Results (%s)' % blob_names[blob])
    plt.xticks(idx+width/2., categories)
    plt.ylim(0.5, 1.0)
    plt.show()

f, ax = plt.subplots(1,2, sharey=True, figsize=(12,4))

cat_idx = range(6)

idx = np.arange(len(cat_idx))
width = 1/float(len(blobs)+1)
counts = np.array([np.sum(pc == i) for i in cat_idx])

j = 0
for blob in blobs:
    lasts = disc_results[blob][:,-1,-1,-1]
    dist_mus = np.array([np.mean([x for x in lasts[pc == i] if not np.isnan(x)]) for i in cat_idx])
    dist_sigmas = np.array([np.std([x for x in lasts[pc == i] if not np.isnan(x)]) for i in cat_idx])
    ax[0].bar(idx + j*width, dist_mus, width, yerr=np.true_divide(dist_sigmas, np.sqrt(counts)), label=blob_names[blob])
    j += 1
#ax.bar(idx+width, mus_single, width, yerr=np.true_divide(sigmas_single, np.sqrt(counts)), label='best filter')
ax[0].legend(loc='lower right')
ax[0].set_ylabel('Classification Accuracy')
ax[0].set_title('Classification Results of by Concept Categories (combination)')
ax[0].set_xticks(idx+width*(len(blobs)-1)/2.)
ax[0].set_xticklabels(categories[cat_idx])
#plt.show()

j = 0
for blob in blobs:
    firsts = disc_results[blob][:,0,-1,-1]
    single_mus = np.array([np.mean([x for x in firsts[pc == i] if not np.isnan(x)]) for i in cat_idx])
    single_sigmas = np.array([np.std([x for x in firsts[pc == i] if not np.isnan(x)]) for i in cat_idx])
    ax[1].bar(idx + j*width, single_mus, width, yerr=np.true_divide(single_sigmas, np.sqrt(counts)), label=blob_names[blob])
    j += 1
#ax.bar(idx+width, mus_single, width, yerr=np.true_divide(sigmas_single, np.sqrt(counts)), label='best filter')
ax[1].legend(loc='upper left')
ax[1].set_ylabel('Classification Accuracy')
ax[1].set_title('Classification Results by Concept Categories (top filter)')
#ax[1].set_yticks(ax[0].get_yticks())
ax[1].set_xticks(idx+width*(len(blobs)-1)/2.)
ax[1].set_xticklabels(categories[cat_idx])

plt.setp(ax[0].get_yticklabels(), visible=True)
plt.setp(ax[1].get_yticklabels(), visible=True)

plt.ylim([0.5,1.0])
formatter = FuncFormatter(to_percent)
plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()
plt.show()

imgs_per_class = np.sum(image_to_label, axis=0)
train_imgs_per_class = np.sum(image_to_label[train_ind,:], axis=0)
val_imgs_per_class = np.sum(image_to_label[val_ind,:], axis=0)

for blob in blobs[2:]:
    f, ax = plt.subplots(1,1)
    x = disc_results[blob][1:,1,-1,-1]
    ax.hist(x[np.isfinite(x)], alpha=0.75, bins=25, label='combo')
    x = disc_results[blob][1:,0,-1,-1]
    ax.hist(x[np.isfinite(x)], alpha=0.75, bins=25, label='top filter')
    ax.set_xlabel('Validation Classification Accuracy')
    ax.set_ylabel('Number of Classes')
    ax.set_title(r'Distribution of Classification Accuracy (%s)' % blob_names[blob]) #$N_{classes}=%d$)' % (blob_names[blob], len(idx)))
    #ax.set_title(r'Distribution of Set IOUs (%s, $N_{classes}=%d, N_{samples} \geq %d, \alpha < %.2f$)' % (blob_names[blob], len(idx), num_samples, alpha_thres))
    ax.legend(loc='upper left')
    #plt.ylim([0.5,1.0])
    plt.show()

perc_factors = []
for blob in blobs:
    x = disc_results[blob][1:,1,-1,-1]
    y = disc_results[blob][1:,0,-1,-1]
    perc_factors.append(np.sum(x[np.isfinite(x)] >= y[np.isfinite(y)])/float(len(x[np.isfinite(x)])))
print perc_factors
print len(x[np.isfinite(x)])
f, ax = plt.subplots(1,1)
ax.bar(np.arange(len(blobs)), perc_factors)
plt.xticks(np.arange(len(blobs)), [blob_names[blob] for blob in blobs])
ax.set_ylabel('Percent combo > top filter')
ax.set_title(r"% of concepts where combo $\geq$ top filter on classification (val acc)")
formatter = FuncFormatter(to_percent)
plt.gca().yaxis.set_major_formatter(formatter)
plt.show()

blob = 'features.7'

x = disc_results[blob][:,1,-1,-1]
y = disc_results[blob][:,0,-1,-1]
label_names[np.where(x < y)[0]]

idx = np.where(np.isfinite(x))[0]
curr_small = np.where(x[idx] < y[idx])[0]
curr_big = np.where(x[idx] >= y[idx])[0]

imgs_per_class = np.sum(image_to_label[train_ind,:], axis=0)

f, ax = plt.subplots(1,1)
ax.hist(np.log10(imgs_per_class[idx[curr_big]]), bins=20, alpha=0.5, label=r'combo $\geq$ best filter')
ax.hist(np.log10(imgs_per_class[idx[curr_small]]), bins=20, alpha=0.5, label='best filter > combo')
#ax.plot(1-alphas[idx[curr_small]], imgs_per_class[curr_small], '.')
ax.legend()
#ax.set_xlabel('Mean fraction of image')
ax.set_ylabel('Number of images in concept class')
#ax.set_title("Mean object size vs. concept dataset size (%s)" % (blob_names[blob]))
plt.show()

print (len(curr_small))
print np.sum(imgs_per_class[idx[curr_small]] <= 20)
print np.sort(imgs_per_class[idx[curr_small]])

f, ax = plt.subplots(1,1)

ranges = [(0,10), (11,15), (16,20), (21,40), (41,80), (81,120),(121, np.inf)]
idx = np.arange(len(ranges))
width = 1/float(len(blobs)+1)

label_idx = np.array([i for i in range(len(pc)) if (pc[i] < 4 and pc[i] >= 0)])

j = 0
for blob in blobs:
    blob_info = ed.load_info(blob=blob)
    K = blob_info.shape[1]
    best_filters = single_set_train_ious[blob].argmax(axis=1)[label_idx]
    best_filters_count = np.zeros(K)
    for f in range(K):
        best_filters_count[f] = np.sum(best_filters == f)
    ax.bar(idx + j*width, [np.sum(np.logical_and(best_filters_count >= r[0], 
                                         best_filters_count <= r[1]))/float(K) for r in ranges], 
           width, label=blob_names[blob])
    j += 1
ax.legend()
ax.set_title('Distribution of Concepts Across Filters (Segmentation)')
ax.set_xticks(idx+width*(len(blobs)-1)/2.)
ax.set_xticklabels(['0','1','2-5','6-9','10-14','15-19','20+'])
ax.set_xlabel('Number of Concepts Encoded by a Filter')
ax.set_ylabel(r'Percent of Filters Encoding $N$ Concepts')
formatter = FuncFormatter(to_percent)
plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()
plt.show()

x = disc_results[blob][:,1,-1,-1]
y = disc_results[blob][:,0,-1,-1]
curr_small = np.where(x[np.isfinite(x)] < y[np.isfinite(y)])[0]
curr_big = np.where(x[np.isfinite(x)] >= y[np.isfinite(y)])[0]
idx = np.where(np.isfinite(x))[0]

f, ax = plt.subplots(1,1)
blob_info = ed.load_info(blob=blob)
filters_range = np.concatenate((num_filters, [blob_info.shape[1]]))
z = disc_results_all[blob][:, :, -1, -1]
ax.errorbar(filters_range, np.mean(z[idx[curr_small]], axis=0), 
            yerr=np.std(z[idx[curr_small]], axis=0)/np.sqrt(z[idx[curr_small]].shape[0]), label='small')
ax.errorbar(filters_range, np.mean(z[idx[curr_big]], axis=0), 
            yerr=np.std(z[idx[curr_big]], axis=0)/np.sqrt(z[idx[curr_big]].shape[0]), label='big')

ax.set_title('Classification Accuracy Using Different # of Filters')
ax.set_xlabel('# Filters')
ax.set_ylabel('One vs. Rest Mean Val Accuracy')
ax.legend()
#plt.ylim([0.5, 1.0])
plt.show()

#idx = np.array([i for i in range(len(pc)) if (pc[i] < 4 and pc[i] >= 0) and i > 0])

label_i = 145

f, ax = plt.subplots(1,1)
ax.plot(disc_weights[blob][label_i,-1,:], weights[blob][label_i], '.')
ax.set_xlabel('Weights from Classification Task')
ax.set_ylabel('Weights from Segmentation Task')
ax.set_title('Relationship between Learned Weights (%s, %s)' % (label_names[label_i], blob_names[blob]))
plt.show()

idx = np.array([i for i in range(len(pc)) if (pc[i] < 4 and pc[i] >= 0)])

rhos = []
pvs = []
for f in range(256):
    (rho, pv) = pearsonr(disc_weights[blob][idx,-1,f], weights[blob][idx,f])
    rhos.append(rho)
    pvs.append(pv)

f, ax = plt.subplots(1,1)
ax.hist(rhos)
plt.show()

np.sum(pvs < 0.01)

idx = np.array([i for i in range(len(pc)) if (pc[i] < 4 and pc[i] >= 0) and i > 0])

rhos = np.zeros(L)
pvs = np.zeros(L)

for label_i in idx:
    (rho, pv) = pearsonr(disc_weights[label_i,-1,:], weights[blob][label_i])
    rhos[label_i] = rho
    pvs[label_i] = pv

f, ax = plt.subplots(1,1)
ax.hist(rhos[rhos > 0], bins=20)
ax.set_xlabel(r'$\rho$ corr. coef between weights from disc and seg tasks')
ax.set_ylabel('# of Concepts')
ax.set_title('Correlation between Learned Weights (pv < 0.01: %.2f=%d/%d, %s)' % (np.sum(pvs[pvs > 0] < 0.01)/float(np.sum(pvs > 0)),
                                                                                 np.sum(pvs[pvs > 0] < 0.01), 
                                                                                 np.sum(pvs > 0), blob_names[blob]))
plt.show()
label_names[1:][x < 0.5]
sel_idx = np.where(x < 0.5)[0]
for i in sel_idx:
    ind = i + 1
    print ind, label_names[ind], imgs_per_class[ind], train_imgs_per_class[ind], val_imgs_per_class[ind], x[i] 
