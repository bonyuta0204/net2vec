import expdir
import loadseg
from upsample_blob_data import get_seg_size
from linearprobe_pytorch import AverageMeter, CustomLayer
from customoptimizer_pytorch import Custom_SGD

from sklearn import svm
#import pickle as pkl
from sklearn.externals import joblib

import time
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import os


def iou_intersect(input, target, target_thresh, threshold=0.5):
    return torch.sum(torch.sum(torch.mul((input > threshold).float(), target), dim=-1),
                     dim=-1)


def iou_union(input, target, threshold=0.5):
    return torch.sum(torch.sum(torch.clamp(torch.add(target,
                                                     (input > threshold).float()),
                                           max=1.), dim=-1), dim=-1)


def BCELoss2d(input, target, alpha=None):
    '''
    Copied from linearprobe_pytorch.py
    '''
    if alpha is None:
        alpha = 1
        beta = 1
    else:
        assert(alpha >= 0 and alpha <= 1)
        beta = 1 - alpha
        return -1./input.size(0) * torch.sum(alpha*torch.mul(target, input) + beta*torch.mul(1-target, 1-input))


def run_epoch(blobdata, conceptdata, example_idx, filter_i, thresh, model, batch_size, criterion, optimizer,
              epoch, train=False, cuda=False, iou_threshold=0.0):
    if train:
        model.train()
        volatile=False
    else:
        model.eval()
        volatile=True

    N = len(example_idx)
    num_batches = int(np.ceil(N/float(batch_size)))
    losses = AverageMeter()
    iou_intersects = np.zeros(N)
    iou_unions = np.zeros(N)
    num_conc = np.zeros(N)
    num_filt_acts = np.zeros(N)

    #no_quantile = not np.any(thresh)
    target_thresh = float(thresh)
    up = nn.Upsample(size=conceptdata.shape[2:], mode='bilinear')

    i = 0
    start = time.time()
    for batch_i in range(100):
        if (i+1)*batch_size < N:
            idx = range(i*batch_size, (i+1)*batch_size)
        else:
            idx = range(i*batch_size, N)
        i += 1

        startt = time.time()
        input = torch.Tensor(conceptdata[example_idx[idx],1:])
        #print 'Time to load concept data:', time.time() - startt
        input_var = (Variable(input.cuda(), volatile=volatile) if cuda
                     else Variable(input, volatile=volatile))
        startt = time.time()
        target = torch.Tensor(blobdata[example_idx[idx],filter_i,np.newaxis]) # new axis needed to do bilinear upsampling
        #print 'Time to load blob data:', time.time() - startt
        #if no_quantile: 
        #    target_var = (up(Variable(target.cuda(), volatile=volatile) if cuda
        #                  else Variable(target, volatile=volatile)).squeeze())
        #else:
        target_var = (up(Variable(target.cuda(), volatile=volatile) if cuda
                      else Variable(target, volatile=volatile)) > target_thresh).float().squeeze()
        output_var = model(input_var)
        loss = criterion(output_var, target_var)
        losses.update(loss.data[0], input.size(0))
        iou_intersects[idx] = iou_intersect(output_var, target_var, iou_threshold
                                            ).data.cpu().numpy()
        iou_unions[idx] = iou_union(output_var, target_var, iou_threshold
                                    ).data.cpu().numpy()
        num_conc[idx] = (output_var > 0.5).float().sum(dim=-1).sum(dim=-1).data.cpu().numpy()
        num_filt_acts[idx] = target_var.sum(dim=-1).sum(dim=-1).data.cpu().numpy()

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        set_iou = np.true_divide(np.sum(iou_intersects[:idx[-1]]), np.sum(iou_unions[:idx[-1]]))
        ind_ious = np.true_divide(iou_intersects, iou_unions)
        concept_given_filter = np.true_divide(np.sum(iou_intersects[:idx[-1]]), np.sum(num_filt_acts[:idx[-1]]))
        filter_given_concept = np.true_divide(np.sum(iou_intersects[:idx[-1]]), np.sum(num_conc[:idx[-1]]))

        if train:
            print('Epoch {}[{}/{}]\t'
                  'Avg Loss {losses.avg:.4f} \t'
                  'Set IoU {}\t'
                  'P(C|F_k) {}\t'
                  'P(F_k|C) {}\t'
                  'Time {}\t'.format(epoch, batch_i+1, num_batches, set_iou, concept_given_filter, 
                      filter_given_concept, time.time()-start, losses=losses))
        else:
            print('Test [{}/{}]\t'
                  'Avg Loss {losses.avg:.4f} \t'
                  'Set IoU {}\t'
                  'p(C|F_k) = {}\t'
                  'p(F_k|C) = {}\t'
                  'Time {}\t'.format(batch_i+1, num_batches, set_iou, concept_given_filter, 
                      filter_given_concept, time.time()-start, losses=losses))

    return (losses.avg, set_iou, ind_ious, concept_given_filter, filter_given_concept)


def reverse_linear_probe(directory, blob, filter_i, suffix='', prev_suffix=None, batch_size=64, quantile=0.005,
                         bias=False, positive=False, use_svm=True, num_examples=1000, num_epochs=30, lr=1e-4, momentum=0.9,
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
        thresh = np.zeros((1,1))
    else:
        quantdata = ed.open_mmap(blob=blob, part='quant-*', shape=(K,-1))
        #threshold = quantdata[:, int(round(quantdata.shape[1] * quantile))]
        threshold = quantdata[filter_i, int(round(quantdata.shape[1] * quantile))]
        #thresh = threshold[:, np.newaxis, np.newaxis]
        thresh = threshold

    blobdata = ed.open_mmap(blob=blob, mode='r', shape=shape)
    #upsampled_data = ed.open_mmap(blob=blob, part='upsampled', mode='r')
    conceptdata = ed.open_mmap(part='concept_data', mode='r',
                               shape=(N, L, Hs, Ws))

    train_idx = np.array([i for i in range(N) if ds.split(i) == 'train'])
    val_idx = np.array([i for i in range(N) if ds.split(i) == 'val'])

    # calculate mean non-negative filter activation
    perc_label = []
    thresh_idx = []
    for i in train_idx:
        num_thresh = np.sum((blobdata[i,filter_i] > thresh).astype(float))
        perc_label.append(num_thresh / float(np.prod(shape[2:])))
        if num_thresh > 0:
            thresh_idx.append(i)
            #perc_label.append(num_nz / float(np.prod(shape[2:])))
    perc_label = np.array(perc_label)
    alpha_unnorm = float(1. - np.mean(perc_label))
    alpha = float(1. - np.mean(perc_label[thresh_idx]))
    train_idx = train_idx[thresh_idx]

    print('Alpha for filter %d: %f (%f unnorm)' % (filter_i, alpha, alpha_unnorm))
    print('# above thresh train examples for filter %d: %d' % (filter_i, len(thresh_idx)))

    if use_svm:
        train_subset_idx = range(num_examples)
        val_subset_idx = range(num_examples)
        start = time.time()
        print 'Selecting training data...'
        X_train = np.mean(np.mean(conceptdata[train_idx[train_subset_idx], :, 52:62, 52:62], axis=-1), axis=-1)
        Y_train = (blobdata[train_idx[train_subset_idx], filter_i, 5, 5] > thresh).astype(int)
        print 'Finished selecting training data in %d secs. Fitting SVM...' % (time.time() - start)
        start = time.time()
        clf = svm.SVC(kernel="linear", class_weight = "balanced")
        clf.fit(X_train, Y_train)
        print 'Finished fitting SVM in %d secs. Selecting validation data...' % (time.time() - start)
        start = time.time()
        X_val = np.mean(np.mean(conceptdata[train_idx[val_subset_idx], :, 52:62, 52:62], axis=-1), axis=-1)
        Y_val = (blobdata[val_idx[val_subset_idx], filter_i, 5, 5] > thresh).astype(float)
        print 'Finished selecting validation data in %d secs. Predicting for validation data...' % (time.time() - start)
        start = time.time()
        Y_pred = clf.predict(X_val)
        print 'Finished predicting validation in %d secs.' % (time.time() - start)
        print 'Val MSE:', np.mean((Y_pred - Y_val)**2)
        joblib.dump(clf, os.path.join(directory, "%s-filter_i_%d_num_examples_%d.pkl" % (blob, filter_i, num_examples)))
    else:
        layer = CustomLayer(num_features=L-1, upsample=False, act=False, positive=positive, bias=bias, cuda=cuda)

        if prev_suffix is not None:
            prev_weights_mmap  = ed.open_mmap(blob=blob, part='filter_i_%d_weights%s' % (filter_i, prev_suffix),
                                    mode='r', shape=layer.weights.shape)
            layer.weights.data[:] = torch.Tensor(prev_weights_mmap[:])
        if cuda:
            layer = layer.cuda()

        #if quantile == 1:
        #    criterion = nn.MSELoss()
        #else:
        #    criterion = nn.BCEWithLogitsLoss()
        #if cuda:
        #    criterion = criterion.cuda()
        criterion = lambda x,y: BCELoss2d(x,y,alpha)

        optimizer = Custom_SGD(layer.parameters(), lr, momentum, l1_weight_decay=l1_weight_decay,
                               l2_weight_decay=l2_weight_decay, nesterov=nesterov, lower_bound=lower_bound)

        results = np.zeros((8,num_epochs))

        for t in range(num_epochs):
            (trn_loss, trn_set_iou, trn_ind_ious, trn_c_given_f, trn_f_given_c) = run_epoch(blobdata, conceptdata, train_idx,
                                                              filter_i, thresh, layer, batch_size, criterion,
                                                              optimizer, t+1, train=True, cuda=cuda,
                                                              iou_threshold=0)
            (val_loss, val_set_iou, val_ind_ious, val_c_given_f, val_f_given_c) = run_epoch(blobdata, conceptdata, val_idx,
                                                              filter_i, thresh, layer, batch_size, criterion,
                                                              optimizer, t+1, train=False, cuda=cuda,
                                                              iou_threshold=0)
            results[0][t] = trn_loss
            results[1][t] = val_loss
            results[2][t] = trn_set_iou
            results[3][t] = val_set_iou
            results[4][t] = trn_c_given_f
            results[5][t] = val_c_given_f
            results[6][t] = trn_f_given_c
            results[7][t] = val_f_given_c

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
            bias_mmap = ed.open_mmap(blob=blob, part='filter_i_%d_bias%s' % (filter_i, suffix),
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
        parser.add_argument('--prev_suffix',
                            type=str,
                            default=None)
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
        parser.add_argument('--use_svm', 
                            action='store_true',
                            default=False)
        parser.add_argument('--num_examples',
                            type=int,
                            default=None)
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
                                     prev_suffix=args.prev_suffix,
                                     batch_size=args.batch_size,
                                     use_svm=args.use_svm,
                                     num_examples=args.num_examples,
                                     quantile=args.quantile,
                                     bias=args.bias,
                                     num_epochs=args.epochs,
                                     lr=args.learning_rate,
                                     cuda=cuda)
    except:
       traceback.print_exc(file=sys.stdout)
       sys.exit(1)



