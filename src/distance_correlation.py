from __future__ import print_function

import expdir
import loadseg

from upsample_blob_data import get_seg_size
from labelprobe import cached_memmap

import torch
import torch.nn as nn
from torch.autograd import Variable 
from torch.nn.parameter import Parameter

import numpy as np
import pickle as pkl
from scipy.spatial.distance import pdist, squareform
import math

from linearprobe_pytorch import AverageMeter

#import timeit
import time
import os


class WeightedData(nn.Module):
    def __init__(self, num_features):
        super(WeightedData, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(self.num_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.num_features)
        self.weight.data.uniform_(-stdv, stdv)
        #self.weight.data[:] = 1

    def forward(self, x):
        return self.weight.unsqueeze(0) * x


def run_epoch(x_data, y_data, x_weights, y_weights, split_idx, criterion, optimizer, epoch, batch_size=64, 
        train=True, randomize=False, cuda=False, verbose=False):
    startt = time.time()
    if train:
        x_weights.train()
        y_weights.train()
        volatile = False
    else:
        x_weights.eval()
        y_weights.eval()
        volatile = True
    
    N = len(split_idx)
    
    num_batches = int(np.ceil(N/float(batch_size)))
    losses = AverageMeter()

    ordered_idx = range(N)
    if randomize:
        ordered_idx = np.random.permutation(ordered_idx)

    i = 0
    start = time.time()
    for batch_i in range(num_batches):
        if (i+1)*batch_size < N:
            idx = split_idx[ordered_idx[range(i*batch_size, (i+1)*batch_size)]]
        else:
            idx = split_idx[ordered_idx[range(i*batch_size, N)]]
       
        if verbose:
            startt = time.time()
            print('Selecting data...')
        x_ = torch.Tensor(x_data[idx])
        y_ = torch.Tensor(y_data[idx])
        if verbose:
            print('Selected data in %d secs.' % (time.time() - startt))

        if verbose:
            startt = time.time()
        x_var = Variable(x_.cuda() if cuda else x_, volatile=volatile) 
        y_var = Variable(y_.cuda() if cuda else y_, volatile=volatile)

        weighted_x = x_weights(x_var)
        weighted_y = y_weights(y_var)
        loss = criterion(weighted_x, weighted_y)
        losses.update(loss.data[0], len(idx)) 

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if train:
        print('Epoch {}[{}/{}]\t'
                'Avg Loss {losses.avg:.4f}\t'
                'Time {}\t'.format(epoch, batch_i+1, num_batches, 
                    time.time() - start, losses=losses))
    else:
        print('Test [{}/{}]\t'
                'Avg Loss {losses.avg:.4f}\t'
                'Time {}\t'.format(batch_i+1, num_batches, 
                    time.time() - start, losses=losses))

    return losses.avg


def pairwise_distance(x, y=None, p=2):
    if y is not None:
        diff = torch.abs(x.unsqueeze(1) - y.unsqueeze(0))
    else:
        diff = torch.abs(x.unsqueeze(1) - x.unsqueeze(0))
    #dist = torch.sum(diff**p, -1)**(1./p) # dist[i][i].grad = nan
    dist = diff.norm(p=p, dim=2)
    return dist


def pairwise_distance_quadratic(x, y=None):
    """
    pairwise_distance(x, y=None)

    Taken from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3

    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^p
    """ 
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = (x_norm + y_norm - 2.0 * torch.mm(x, y_t))**0.5
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    #dist = torch.clamp(dist, 0.0, np.inf)
    dist[dist != dist] = 0 # replace nan values with 0
    return dist


#grads = {}
#def save_grad(name):
#    def hook(grad):
#        print(name, grad.size(), grad[0][:5])
#        grads[name] = grad 
#    return hook


"""
Inspired by https://gist.github.com/satra/aa3d19a12b74e9ab7941
"""
def distcorr(x, y, use_quad=False):
    # TODO: Do more input checking
    n = x.size(0)
    if use_quad:
        a = pairwise_distance_quadratic(x)
        b = pairwise_distance_quadratic(y)
    else:
        a = pairwise_distance(x, p=2)
        b = pairwise_distance(y, p=2)
    A = a - a.mean(dim=0).unsqueeze(0) - a.mean(dim=1).unsqueeze(1) + a.mean()
    B = b - b.mean(dim=0).unsqueeze(0) - b.mean(dim=1).unsqueeze(1) + b.mean()

    dcov2_xy = (A * B).sum()/float(n**2)
    dcov2_xx = (A * A).sum()/float(n**2)
    dcov2_yy = (B * B).sum()/float(n**2)
    dcor = dcov2_xy**0.5/(dcov2_xx**0.5 * dcov2_yy**0.5)**0.5
    return dcor


"""
For testing
"""
def distcorr_numpy(x, y):
    n = x.shape[0]
    a = squareform(pdist(x, p=2))
    b = squareform(pdist(y, p=2))
    A = a - a.mean(axis=0)[None,:] - a.mean(axis=1)[:,None] + a.mean()
    B = b - b.mean(axis=0)[None,:] - b.mean(axis=1)[:,None] + b.mean()

    dcov2_xy = (A * B).sum()/float(n**2)
    dcov2_xx = (A * A).sum()/float(n**2)
    dcov2_yy = (B * B).sum()/float(n**2)
    dcor = dcov2_xy**0.5/(dcov2_xx**0.5 * dcov2_yy**0.5)**0.5
    return dcor
    

def test(N=1000, d=500, cuda=False):
    N = 1000
    d = 500
    x = np.random.rand(N,d)
    y = np.random.rand(N,d)
    
    dc_numpy = distcorr_numpy(x,y)
    xv = Variable(torch.from_numpy(x).cuda() if cuda else torch.from_numpy(x))
    yv = Variable(torch.from_numpy(y).cuda() if cuda else torch.from_numpy(y))
    start = time.time()
    dc_pytorch = distcorr(xv,yv, use_quad=False)
    print(time.time()-start)
    start = time.time()
    dc_pytorch_quad = distcorr(xv,yv, use_quad=True)
    print(time.time()-start)

    #print pairwise_distance(xv)
    #print pairwise_distance_quadratic(xv)
    #print dc_pytorch
    #print dc_pytorch_quad

    #def wrap_no_quad():
    #    return distcorr(xv, yv, use_quad=False)
    
    #def wrap_quad():
    #    return distcorr(xv, yv, use_quad=True)

    #print timeit.timeit(wrap_no_quad)
    #print timeit.timeit(wrap_quad)

    assert(np.isclose(dc_numpy, dc_pytorch.data.cpu().numpy()[0]))
    assert(np.isclose(dc_numpy, dc_pytorch_quad.data.cpu().numpy()[0]))


def compute_distance_correlation(directory, blob, out_file=None, learning_rate=1e-4, momentum=0.9, nesterov=False, 
        num_epochs=10, batch_size=128, randomize=True, cuda=False, verbose=False):
    ed = expdir.ExperimentDirectory(directory)

    info = ed.load_info()
    ds = loadseg.SegmentationData(info.dataset)

    L = ds.label_size()
    N = ds.size()

    blob_info = ed.load_info(blob=blob)
    K = blob_info.shape[1]

    categories = np.array(ds.category_names())
    label_names = np.array([ds.label[i]['name'] for i in range(L)])

    (Hs, Ws) = get_seg_size(info.input_dim)

    #concept_data = ed.open_mmap(part='concept_data', mode='r',
    #        shape=(N,L,Hs,Ws))
    #upsampled_data = ed.open_mmap(blob=blob, part='upsampled', mode='r',
    #        shape=(N,K,Hs,Ws))
    concept_fn = ed.mmap_filename(part='concept_data')
    upsampled_fn = ed.mmap_filename(blob=blob, part='upsampled')
    concept_data = cached_memmap(concept_fn, mode='r', dtype='float32', shape=(N,L,Hs,Ws))
    upsampled_data = cached_memmap(upsampled_fn, mode='r', dtype='float32', shape=(N,K,Hs,Ws))

    concept_weights = WeightedData(L)
    act_weights = WeightedData(K)

    if cuda:
        concept_weights = concept_weights.cuda()
        act_weights = act_weights.cuda()

    criterion = lambda x,y: -1*distcorr(x,y,use_quad=False)
    optimizer = torch.optim.SGD([{'params': concept_weights.parameters()}, {'params': act_weights.parameters()}], 
            lr=args.learning_rate, momentum=args.momentum, nesterov=args.nesterov)

    train_idx = np.array([i for i in range(N) if ds.split(i) == 'train'])
    val_idx = np.array([i for i in range(N) if ds.split(i) == 'val'])

    train_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    for t in range(num_epochs):
        train_loss = run_epoch(concept_data[:,:,Hs/2,Ws/2], upsampled_data[:,:,Hs/2,Ws/2], concept_weights, 
                act_weights, train_idx, criterion, optimizer, t+1, batch_size=batch_size, 
                train=True, randomize=randomize, cuda=cuda, verbose=verbose)
        val_loss = run_epoch(concept_data[:,:,Hs/2,Ws/2], upsampled_data[:,:,Hs/2,Ws/2], concept_weights, 
                act_weights, val_idx, criterion, optimizer, t+1, batch_size=batch_size, 
                train=False, randomize=randomize, cuda=cuda, verbose=verbose)
        train_losses[t] = train_loss
        val_losses[t] = val_loss

    #print(distcorr(x_weights(Variable(torch.Tensor(x).cuda())),
    #    y_weights(Variable(torch.Tensor(y).cuda())).data.cpu().numpy()))
    print(concept_weights.weight)
    print(act_weights.weight)
    
    results = {}
    if out_file is not None:
        if verbose:
            start = time.time()
            print('Saving results...')
        results['directory'] = directory
        results['blob'] = blob
        results['learning_rate'] = learning_rate
        results['momentum'] = momentum
        results['nesterov'] = nesterov
        results['batch_size'] = batch_size
        results['train_losses'] = train_losses
        results['val_losses'] = val_losses
        results['num_epochs'] = num_epochs
        results['randomize'] = randomize
        results['concept_weights'] = concept_weights.weight.data.cpu().numpy()
        results['act_weights'] = act_weights.weight.data.cpu().numpy()

        pkl.dump(results, open(out_file, 'wb'))
        if verbose:
            print('Saved results at %s in %d secs.' % (out_file, time.time() - start))


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser(description='TODO')
        parser.add_argument('--directory', 
                type=str)
        parser.add_argument('--blobs',
                type=str,
                nargs='*',)
        parser.add_argument('--gpu',
                type=int,
                default=None)
        #parser.add_argument('-N', '--num_examples',
        #        type=int,
        #        default=1000)
        #parser.add_argument('-K', '--num_features',
        #        type=int,
        #        default=5)
        parser.add_argument('--learning_rate',
                type=float,
                default=1e-4)
        parser.add_argument('--momentum',
                type=float,
                default=0.9)
        parser.add_argument('--nesterov',
                action='store_true',
                default=False)
        parser.add_argument('-E', '--num_epochs',
                type=int,
                default=10)
        parser.add_argument('-B', '--batch_size',
                type=int,
                default=128)
        parser.add_argument('--randomize',
                action='store_true',
                default=False)
        parser.add_argument('-O', '--out_file',
                type=str,
                default=None)

        args = parser.parse_args()

        gpu = args.gpu 
        cuda = True if gpu is not None else False
        use_mult_gpu = isinstance(gpu, list)
        if cuda:
            if use_mult_gpu:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu).strip('[').strip(']')
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpu

        for blob in args.blobs:
            compute_distance_correlation(args.directory, blob, out_file=args.out_file, 
                    learning_rate=args.learning_rate, momentum=args.momentum, 
                    nesterov=args.nesterov, num_epochs=args.num_epochs, 
                    batch_size=args.batch_size, cuda=cuda, verbose=False)

    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)

def test2(cuda=False):
    N = args.num_examples
    K = args.num_features
    x = np.random.randn(N, K)
    y = np.zeros((N, K))

    for k in range(K):
        y[:,k] = x[:,k]**2 #+ 0.1*np.random.rand(N) 

    print(distcorr_numpy(x,y))

    for k in range(K):
        y[:,k] = x[:,k]**2 + 1.*np.random.rand(N)

    print(distcorr_numpy(x,y))


    x_weights = WeightedData(K)
    y_weights = WeightedData(K)

    if cuda:
        x_weights = x_weights.cuda()
        y_weights = y_weights.cuda()

    criterion = lambda x,y: -1*distcorr(x,y,use_quad=False)
    #criterion = nn.MSELoss()
    #optimizer = torch.optim.SGD([{'params':x_weights.parameters()}, {'params':y_weights.parameters()}], 
    #        lr=args.learning_rate, momentum=args.momentum, nesterov=args.nesterov)
    optimizer = torch.optim.SGD(x_weights.parameters(), 
            lr=args.learning_rate, momentum=args.momentum, nesterov=args.nesterov)

    for t in range(args.num_epochs):
        run_epoch(x, y, x_weights, y_weights, criterion, optimizer, t+1, batch_size=args.batch_size, 
                train=True, cuda=cuda)

    print(distcorr_numpy(x_weights(Variable(torch.Tensor(x).cuda())).data.cpu().numpy(),
        y_weights(Variable(torch.Tensor(y).cuda())).data.cpu().numpy()))
    print(x_weights.weight)
    print(y_weights.weight)
