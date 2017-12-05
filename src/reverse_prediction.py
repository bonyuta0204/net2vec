import expdir
import loadseg
from upsample_blob_data import get_seg_size
from linearprobe_pytorch import AverageMeter, CustomLayer
from customoptimizer_pytorch import Custom_SGD

import time
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import os


def iou_intersect(input, target, target_thresh, threshold=0.5):
    return torch.sum(torch.sum(torch.mul((input > threshold).float(),
                                         (target > target_thresh).float()), dim=-1),
                     dim=-1)


def iou_union(input, target, target_thresh, threshold=0.5):
    return torch.sum(torch.sum(torch.clamp(torch.add((target > target_thresh).float(),
                                                     (input > threshold).float()),
                                           max=1.), dim=-1), dim=-1)


def run_epoch(blobdata, conceptdata, example_idx, filter_i, thresh, model, batch_size, criterion, optimizer,
              epoch, train=False, cuda=False, iou_threshold=0.5):
    if train:
        model.train()
        volatile=False
    else:
        model.eval()
        volatile=True

    N = len(example_idx)
    num_batches = int(np.ceil(N/float(batch_size)))
    losses = AverageMeter()
    iou_intersects = []
    iou_unions = []

    target_thresh = float(thresh[filter_i])
    up = nn.Upsample(size=conceptdata.shape[2:], mode='bilinear')

    i = 0
    start = time.time()
    for batch_i in range(num_batches):
        if (i+1)*batch_size < N:
            idx = range(i*batch_size, (i+1)*batch_size)
        else:
            idx = range(i*batch_size, N)
        i += 1

        startt = time.time()
        input = torch.Tensor(conceptdata[example_idx[idx],1:])
        print 'Time to load concept data:', time.time() - startt
        input_var = (Variable(input.cuda(), volatile=volatile) if cuda
                     else Variable(input, volatile=volatile))
        startt = time.time()
        target = torch.Tensor(blobdata[example_idx[idx],filter_i,np.newaxis])
        print 'Time to load blob data:', time.time() - startt
        target_var = (up(Variable(target.cuda(), volatile=volatile) if cuda
                      else Variable(target, volatile=volatile)) > target_thresh).float().squeeze()
        output_var = model(input_var)
        loss = criterion(output_var, target_var)
        print loss
        losses.update(loss.data[0], input.size(0))
        iou_intersects.extend(iou_intersect(output_var, target_var, target_thresh, iou_threshold
                                            ).data.cpu().numpy())
        iou_unions.extend(iou_union(output_var, target_var, target_thresh, iou_threshold
                                    ).data.cpu().numpy())
        print 'h'
        if train:
            print 'h1'
            optimizer.zero_grad()
            print 'h2'
            loss.backward()
            print 'h3'
            optimizer.step()
        print 'h4'

        set_iou = np.true_divide(np.sum(iou_intersects), np.sum(iou_unions))
        ind_ious = np.true_divide(iou_intersects, iou_unions)

        if train:
            print('Epoch {0}\t'
                  'Avg Loss {losses.avg:.4f} \t'
                  'Set IoU {1}\t'
                  'Time {2}\t'.format(epoch, set_iou, time.time()-start, losses=losses))
        else:
            print('Test\t'
                  'Avg Loss {losses.avg:.4f} \t'
                  'Set IoU {0}\t'
                  'Time {1}\t'.format(set_iou, time.time()-start, losses=losses))

        return (losses.avg, set_iou, ind_ious)


def reverse_linear_probe(directory, blob, filter_i, suffix='', batch_size=64, quantile=0.005,
                         bias=False, positive=False, num_epochs=30, lr=1e-4, momentum=0.9,
                         l1_weight_decay=0, l2_weight_decay=0, validation=False, nesterov=False,
                         lower_bound=None, cuda=False):
    ed = expdir.ExperimentDirectory(directory)
    info = ed.load_info()
    ds = loadseg.SegmentationData(info.dataset)
    L = ds.label_size()
    N = ds.size()
    (Hs, Ws) = get_seg_size(info.input_dim)
    blob_info = ed.load_info(blob=blob)
    shape = blob_info.shape
    K = shape[1]

    if quantile == 1:
        thresh = None
    else:
        quantdata = ed.open_mmap(blob=blob, part='quant-*', shape=(K,-1))
        threshold = quantdata[:, int(round(quantdata.shape[1] * quantile))]
        #thresh = threshold[:, np.newaxis, np.newaxis]
        thresh = threshold

    blobdata = ed.open_mmap(blob=blob, mode='r', shape=shape)
    #upsampled_data = ed.open_mmap(blob=blob, part='upsampled', mode='r')
    conceptdata = ed.open_mmap(part='concept_data', mode='r',
                               shape=(N, L, Hs, Ws))

    train_idx = np.array([i for i in range(N) if ds.split(i) == 'train'])
    val_idx = np.array([i for i in range(N) if ds.split(i) == 'val'])

    layer = CustomLayer(num_features=L-1, upsample=False, act=False, positive=positive, bias=bias)
    if cuda: # TODO: Debug why this isn't working on mistborn :(
        layer.cuda()

    #criterion = nn.MSELoss()
    criterion = nn.BCEWithLogitsLoss()

    optimizer = Custom_SGD(layer.parameters(), lr, momentum, l1_weight_decay=l1_weight_decay,
                           l2_weight_decay=l2_weight_decay, nesterov=nesterov, lower_bound=lower_bound)

    results = np.zeros((4,num_epochs))

    for t in range(num_epochs):
        (trn_loss, trn_set_iou, trn_ind_ious) = run_epoch(blobdata, conceptdata, train_idx,
                                                          filter_i, thresh, layer, batch_size, criterion,
                                                          optimizer, t+1, train=True, cuda=cuda,
                                                          iou_threshold=0.5)
        (val_loss, val_set_iou, val_ind_ious) = run_epoch(blobdata, conceptdata, val_idx,
                                                          filter_i, thresh, layer, batch_size, criterion,
                                                          optimizer, t+1, train=True, cuda=cuda,
                                                          iou_threshold=0.5)
        results[0][t] = trn_loss
        results[1][t] = val_loss
        results[2][t] = trn_set_iou
        results[3][t] = val_set_iou

    ind_ious = np.zeros(N)
    ind_ious[train_idx] = trn_ind_ious
    ind_ious[val_idx] = val_ind_ious

    ind_ious_mmap = ed.open_mmap(blob=blob, part='filter_i_%d_ind_ious%s' % (filter_i, suffix),
                                 mode='w+', shape=ind_ious.shape)
    ind_ious_mmap[:] = ind_ious[:]
    ed.finish_mmap(ind_ious_mmap)

    results_mmap = ed.open_mmap(blob=blob, part='filter_i_%d_results%s' % (filter_i, suffix),
                               mode='w+', shape=results.shape)
    results_mmap[:] = results[:]
    ed.finish_mmap(results_mmap)

    # save learned weights (and bias)
    weights = layer.weight.data.cpu().numpy()
    weights_mmap = ed.open_mmap(blob=blob, part='filter_i_%d_weights%s' % (filter_i, suffix),
                                mode='w+', shape=weights.shape)
    weights_mmap[:] = weights[:]
    ed.finish_mmap(weights_mmap)

    if bias:
        bias_v = layer.bias.data.cpu().numpy()
        bias_mmap = ed.open_mmap(blob=blob, part='filter_i_%d_weights%s' % (filter_i, suffix),
                                 mode='w+', shape=(1,))
        bias_mmap[:] = bias_v[:]
        ed.finish_mmap(bias_mmap)

if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser(description='TODO')
        parser.add_argument('--directory')
        parser.add_argument('--blobs',
                            nargs='*',
                            type=str)
        parser.add_argument('--filters',
                            nargs='*',
                            type=int,
                            default=None)
        parser.add_argument('--start',
                            type=int,
                            default=0)
        parser.add_argument('--end',
                            type=int,
                            default=None)
        parser.add_argument('--suffix',
                            type=str,
                            default='')
        parser.add_argument('--batch_size',
                            type=int,
                            default=64)
        parser.add_argument('--quantile',
                            type=float,
                            default=0.005)
        parser.add_argument('--learning_rate',
                            type=float,
                            default=1e-4)
        parser.add_argument('--epochs',
                            type=int,
                            default=30)
        parser.add_argument('--bias',
                            action='store_true',
                            default=False)
        parser.add_argument('--gpu',
                            type=int,
                            default=None)

        args = parser.parse_args()

        if args.filters is not None:
            filters = args.filters
        else:
            if args.end is not None:
                filters = range(args.start, args.end)
            else:
                assert(False) # TODO implement

        gpu = args.gpu
        cuda = True if gpu is not None else False
        use_mult_gpu = isinstance(gpu, list)
        if cuda:
            if use_mult_gpu:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu).strip('[').strip(']')
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpu
        print torch.cuda.device_count(), use_mult_gpu, cuda

        for blob in args.blobs:
            for filter_i in filters:
                reverse_linear_probe(args.directory, blob, filter_i,
                                     suffix=args.suffix,
                                     batch_size=args.batch_size,
                                     quantile=args.quantile,
                                     bias=args.bias,
                                     num_epochs=args.epochs,
                                     lr=args.learning_rate,
                                     cuda=cuda)
    except:
       traceback.print_exc(file=sys.stdout)
       sys.exit(1)



