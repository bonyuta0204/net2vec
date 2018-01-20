import loadseg
import expdir
import upsample

from labelprobe import onehot, primary_categories_per_index, cached_memmap
from indexdata import has_image_to_label, load_image_to_label, create_image_to_label

import os
import time
import math

from customoptimizer_pytorch import Custom_SGD

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomLayer(nn.Module):
    def __init__(self, num_features, upsample=True, up_size=(224,224), 
            mask_idx=None, act=True, positive=False, bias=False, cuda=False):
        super(CustomLayer, self).__init__()
        self.num_features = num_features
        self.positive = positive
        self.weight = Parameter(torch.Tensor(self.num_features))
        mask_ = torch.ones(self.num_features)
        if mask_idx is not None:
            mask_[:] = 0
            mask_[mask_idx] = 1
        self.mask = Variable(mask_.cuda() if cuda else mask_)
        if bias:
            self.bias = Parameter(torch.Tensor(1))
        else:
            self.bias = 0
        self.up = upsample
        if self.up:
            self.upsample = nn.Upsample(size=up_size, mode='bilinear')
        self.act = act
        if act:
            self.activation = nn.Sigmoid() #nn.LogSigmoid()
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.num_features)
        if self.positive:
            #self.weight.data.uniform_(0, stdv/2)
            self.weight.data.uniform_(stdv/2, stdv/2)
        else:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        y = x * (self.mask * self.weight).unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        #print(y.size(), y.sum(1).size(), self.bias.size(), (y.sum(1) + self.bias).size())
        if self.act:
            if self.up:
                return self.activation(self.upsample((y.sum(1) + self.bias).unsqueeze(1)).squeeze())
            else:
                return self.activation((y.sum(1) + self.bias).squeeze())
        else:
            if self.up:
                return self.upsample((y.sum(1) + self.bias).unsqueeze(1)).squeeze()
            else:
                return (y.sum(1) + self.bias).squeeze()


def BCELoss2d(input, target, alpha=None):
    if alpha is None:
        alpha = 1
        beta = 1
    else:
        assert(alpha >= 0 and alpha <= 1)
        beta = 1 - alpha
    return -1./input.size(0) * torch.sum(alpha*torch.mul(target, input) + beta*torch.mul(1-target, 1-input))


def iou_intersect(input, target, threshold = 0.5):
    return torch.sum(torch.mul((input > threshold).float(), target))


def iou_union(input, target, threshold = 0.5):
    return torch.sum(torch.clamp(torch.add(target, (input > threshold).float()), max=1.))


def run_epoch(activations, act_idx, label_categories, label_i, fieldmap, thresh, sh, sw, 
        reduction, loader, model, criterion, optimizer, epoch, train=True, 
        cuda=False, iou_threshold=0.5):
    if train:
        model.train()
        volatile=False
    else:
        model.eval()
        volatile=True

    batch_size = loader.batch_size
    N = len(act_idx)

    losses = AverageMeter()
    iou_intersects = AverageMeter()
    iou_unions = AverageMeter()

    i = 0
    start = time.time()
    for batch in loader.batches():
        #start = time.time()
        if (i+1)*batch_size < N:
            idx = range(i*batch_size, (i+1)*batch_size)
        else:
            idx = range(i*batch_size, N)
        i += 1
        #up = [upsample.upsampleL(fieldmap, act, shape=(sh,sw), reduction=reduction)
        #        for act in activations[idx]]
        #input = torch.Tensor((up > thresh).astype(float))
        input = torch.Tensor((activations[act_idx[idx]] > thresh).astype(float))
        input_var = (Variable(input.cuda(), volatile=volatile) if cuda
                else Variable(input, volatile=volatile))
        target = torch.Tensor([np.max((rec[label_categories[idx[j]]] 
                == label_i).astype(float), axis=0) 
                if type(rec[label_categories[idx[j]]]) is np.ndarray 
                else np.ones((sh, sw))
                for j, rec in enumerate(batch)]) 
        target_var = (Variable(target.cuda(), volatile=volatile) if cuda
                else Variable(target, volatile=volatile))
        output_var = model(input_var)
        loss = criterion(output_var, target_var)
        losses.update(loss.data[0], input.size(0))
        iou_intersects.update(iou_intersect(output_var, target_var, iou_threshold
            ).data.cpu().numpy())
        iou_unions.update(iou_union(output_var, target_var, iou_threshold
            ).data.cpu().numpy())

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        iou = np.true_divide(iou_intersects.sum, iou_unions.sum)[0]

        #if train:
        #    print('Epoch {0}[{1}/{2}]\t'
        #          'Avg Loss {losses.avg:.4f}\t'
        #          'Overall IOU {3}\t'
        #          'Time {4}\t'.format(epoch, i, int(round(N/batch_size)), 
        #              iou, time.time()-start, losses=losses))
        #else:
        #    print('Test [{0}/{1}]\t'
        #          'Avg Loss {losses.avg:.4f}\t'
        #          'Overall IOU {2}\t'
        #          'Time {3}\t'.format(i, int(round(N/batch_size)), 
        #              iou, time.time()-start, losses=losses))

    if train:
        print('Epoch {0}\t'
              'Avg Loss {losses.avg:.4f}\t'
              'Overall IOU {1}\t'
              'Time {2}\t'.format(epoch, iou, time.time()-start, losses=losses))
    else:
        print('Test\t'
              'Avg Loss {losses.avg:.4f}\t'
              'Overall IOU {0}\t'
              'Time {1}\t'.format(iou, time.time()-start, losses=losses))

    return (losses.avg, iou)


def linear_probe(directory, blob, label_i, suffix='', init_suffix='', num_filters=None, batch_size=16, ahead=4, 
        quantile=0.005, bias=False, positive=False, num_epochs=30, lr=1e-4, momentum=0.9, 
        l1_weight_decay=0, l2_weight_decay=0, validation=False, nesterov=False, lower_bound=None,
        min_train=None, max_train=None, max_val=None,
        cuda=False):
    # Make sure we have a directory to work in
    #qcode = ('%f' % quantile).replace('0.','').rstrip('0')
    ed = expdir.ExperimentDirectory(directory)
    # Check if linear weights have already been learned 
    print ed.mmap_filename(blob=blob, part='label_i_%s_weights%s' % (label_i, suffix))
    if ed.has_mmap(blob=blob, part='label_i_%d_weights%s' % (label_i, suffix)):
        print('%s already has %s, so skipping.' % (directory,
            ed.mmap_filename(blob=blob, part='label_i_%d_weights%s' % (label_i, suffix))))
        return
    # Load probe metadata
    info = ed.load_info()
    ih, iw = info.input_dim
    # Load blob metadata
    blob_info = ed.load_info(blob=blob)
    shape = blob_info.shape
    unit_size = shape[1]
    fieldmap = blob_info.fieldmap
    # Load the blob quantile data and grab thresholds
    if quantile == 1:
        thresh = np.zeros((unit_size,1,1))
    else:
        quantdata = ed.open_mmap(blob=blob, part='quant-*', shape=(unit_size, -1))
        threshold = quantdata[:, int(round(quantdata.shape[1] * quantile))]
        thresh = threshold[:, np.newaxis, np.newaxis]
    #print np.max(thresh), thresh.shape, type(thresh)
    # Map the blob activation data for reading
    fn_read = ed.mmap_filename(blob=blob)
    # Load the dataset
    ds = loadseg.SegmentationData(info.dataset)
    # Get all the categories the label is a part of
    label_categories = ds.label[label_i]['category'].keys()
    num_categories = len(label_categories)
    # Get label name
    label_name = ds.name(category=None, j=label_i)

    blobdata = cached_memmap(fn_read, mode='r', dtype='float32', shape=shape)
    # Get indices of images containing the given label
    if not has_image_to_label(directory):
        print('image_to_label does not exist in %s; creating it now...' % directory)
        create_image_to_label(directory, batch_size=batch_size, ahead=ahead)
    image_to_label = load_image_to_label(directory)
    label_idx = np.where(image_to_label[:, label_i])[0]
    train_idx = np.array([i for i in label_idx if ds.split(i) == 'train'])
    val_idx = np.array([i for i in label_idx if ds.split(i) == 'val'])
    if min_train is not None and len(train_idx) < min_train:
        print('Number of training examples for label %d (%s) is %d, which is less than the minimum of %d so skipping.' 
                % (label_i, label_name, len(train_idx), min_train))
    if max_train is not None and len(train_idx) > max_train:
        train_idx = train_idx[:max_train]
    if max_val is not None and len(val_idx) > max_val:
        val_idx = val_idx[:max_val]

    print('Total number of images containing label %d (%s): %d' % (
        label_i, label_name, len(label_idx)))
    
    try:
        train_loader = loadseg.SegmentationPrefetcher(ds, categories=label_categories,
                                                      indexes=train_idx, once=False,
                                                      batch_size=batch_size,
                                                      ahead=ahead, thread=True)
    except IndexError as err:
        print(err.args)
        return
    
    sw = 0
    sh = 0
    perc_label = []
    train_label_categories = []
    for batch in train_loader.batches():
        for rec in batch:
            # Check that the same segmentation dimensions are used for all
            # examples
            sw_r, sh_r = [rec[k] for k in ['sw', 'sh']]
            if sw == 0 and sh == 0:
                sw = sw_r
                sh = sh_r
            else:
                assert(sw == sw_r and sh == sh_r)
            for cat in label_categories:
                if rec[cat] != []:
                    train_label_categories.append(cat)
                    if type(rec[cat]) is np.ndarray:
                        perc_label.append(np.sum(rec[cat] == label_i) / float(sw * sh))
                    else:
                        perc_label.append(1.)
                    break
    assert(len(perc_label) == len(train_idx))

    # Compute reduction from segmentation dimensions to image dimensions
    reduction = int(round(iw / float(sw)))
    # Calculate class-weighting alpha parameter for segmentation loss
    # (Note: float typecast is necessary)
    alpha = float(1. - np.mean(perc_label))
    if alpha == 0:
        alpha = None
        print('Not using class-weighting because no pixel-level annotations')
    else:
        print('Alpha for label %d (%s): %f' % (label_i, label_name, alpha))

    # Prepare segmentation loss function using class-weight alpha
    criterion = lambda x,y: BCELoss2d(x,y,alpha)
    # Prepare to learn linear weights with a sigmoid activation after
    # the linear layer
    #layer = CustomLayer(unit_size, upsample=False, act=True, positive=False)
    if num_filters is not None:
        if ed.has_mmap(blob=blob, part='label_i_%d_weights%s' % (label_i, init_suffix)):
            init_weights_mmap = ed.open_mmap(blob=blob, part='label_i_%d_weights%s' % (label_i, init_suffix), 
                    mode='r', dtype='float32', shape=unit_size)
        elif ed.has_mmap(blob=blob, part='linear_weights%s' % (init_suffix)):
            all_weights_mmap = ed.open_mmap(blob=blob, part='linear_weights%s' % init_suffix,
                    mode='r', dtype='float32', shape=(ds.label_size(),unit_size))
            init_weights_mmap = all_weights_mmap[label_i]
        else:
            assert(False)
        sorted_idx = np.argsort(np.abs(init_weights_mmap))[::-1]
        mask_idx = np.zeros(unit_size, dtype=int)
        mask_idx[sorted_idx[:num_filters]] = 1
        layer = CustomLayer(unit_size, upsample=True, up_size=(sh,sw), act=True,
                bias=bias, positive=positive, mask_idx=torch.ByteTensor(mask_idx), cuda=cuda)
    else:
        layer = CustomLayer(unit_size, upsample=True, up_size=(sh,sw), act=True, 
                bias=bias, positive=positive, cuda=cuda)
    if cuda:
        layer.cuda()

    optimizer = Custom_SGD(layer.parameters(), lr, momentum,
            l1_weight_decay=l1_weight_decay, l2_weight_decay=l2_weight_decay,
            nesterov=nesterov, lower_bound=lower_bound)

    if not validation:
        try:
            val_loader = loadseg.SegmentationPrefetcher(ds, categories=label_categories,
                    indexes=val_idx, once=False, batch_size=batch_size,
                    ahead=ahead, thread=True)
        except IndexError as err:
            print(err.args)
            train_loader.close()
            return

        val_label_categories = []
        for batch in val_loader.batches():
            for rec in batch:
                for cat in label_categories:
                    if rec[cat] != []:
                        val_label_categories.append(cat)
                        break
        assert(len(val_label_categories) == len(val_idx))

    for t in range(num_epochs):
        (_, iou) = run_epoch(blobdata, train_idx, train_label_categories, label_i,
                fieldmap, thresh, sh, sw, reduction, train_loader, layer, criterion, 
                optimizer, t+1, train=True, cuda=cuda, iou_threshold=0.5)
        if not validation:
            (_, iou) = run_epoch(blobdata, val_idx, val_label_categories, label_i,
                    fieldmap, thresh, sh, sw, reduction, val_loader, layer, criterion,
                    optimizer, t+1, train=False, cuda=cuda, iou_threshold=0.5)

    # Close segmentation prefetcher (i.e. close pools)
    train_loader.close()
    if not validation:
        val_loader.close()

    # Save weights
    weights = (layer.mask * layer.weight).data.cpu().numpy()
    weights_mmap = ed.open_mmap(blob=blob, part='label_i_%d_weights%s' % (label_i, suffix),
            mode='w+', dtype='float32', shape=weights.shape)
    weights_mmap[:] = weights[:]
    ed.finish_mmap(weights_mmap)
    if bias:
        bias_v = layer.bias.data.cpu().numpy()
        bias_mmap = ed.open_mmap(blob=blob, part='label_i_%d_bias%s' % (label_i, suffix),
                mode='w+', dtype='float32', shape=(1,))
        bias_mmap[:] = bias_v[:]
        ed.finish_mmap(bias_mmap)
    print '%s finished' % ed.mmap_filename(blob=blob, part='label_i_%d_weights%s' % (label_i, suffix))


if __name__ == '__main__':
    import argparse
    import sys
    import traceback
    
    try:
        parser = argparse.ArgumentParser(
                description='Learn linear weights for maximizing segmentation loss\
                        for a given label')
        parser.add_argument(
                '--directory',
                default='.',
                help='output directory for the net probe')
        parser.add_argument(
                '--blobs',
                nargs='*',
                help='network blob names to tally')
        parser.add_argument(
                '--labels',
                nargs='*',
                help='class label indexes')
        parser.add_argument(
                '--start',
                type=int,
                default=1,
                help='start index for class label (inclusive)')
        parser.add_argument(
                '--end',
                type=int,
                default=1198,
                help='end index for class label (exclusive)')
        parser.add_argument(
                '--suffix',
                type=str,
                default='',
                help='TODO')
        parser.add_argument(
                '--batch_size',
                type=int, 
                default=16,
                help='the batch size to use')
        parser.add_argument(
                '--ahead',
                type=int,
                default=4,
                help='the prefetch lookahead size')
        parser.add_argument(
                '--quantile',
                type=float,
                default=0.005,
                help='the quantile cutoff to use')
        parser.add_argument(
                '--learning_rate',
                type=float,
                default=1e-4,
                help='the learning rate for SGD')
        parser.add_argument(
                '--num_epochs',
                type=int,
                default=30,
                help='the number of epochs to train for')
        parser.add_argument(
                '--lower_bound',
                type=int,
                default=None,
                help='TODO')
        parser.add_argument(
                '--bias', 
                action='store_true',
                default=False,
                help='TODO')
        parser.add_argument(
                '--positive',
                action='store_true',
                default=False,
                help='TODO')
        parser.add_argument(
                 '--gpu',
                type=int,
                default=None,
                help='use GPU for training')
        parser.add_argument(
                '--l1_decay',
                type=float,
                default=0,
                help='L1 weight decay hyperparameter')
        parser.add_argument(
                '--l2_decay',
                type=float,
                default=0,
                help='L2 weight decay hyperparameter')
        parser.add_argument(
                '--validation',
                action='store_true',
                default=False,
                help='Train on the validation set (default: train only on training set)')
        parser.add_argument(
                '--num_filters',
                type=int,
                nargs='*',
                default=None,
                help="TODO")
        parser.add_argument(
                '--init_suffix',
                type=str,
                default='',
                help='TODO')
        parser.add_argument(
                '--min_train',
                type=int,
                default=None,
                help='TODO')
        parser.add_argument(
                '--max_train',
                type=int,
                default=None,
                help='TODO')
        parser.add_argument(
                '--max_val',
                type=int,
                default=None,
                help='TODO')

        args = parser.parse_args()
        if args.labels is not None:
            labels = args.labels
        else:
            labels = range(args.start, args.end)

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
            for label_i in labels:
                if args.num_filters is not None:
                    suffix = ['%s_num_filters_%d' % (args.suffix, n) for n in args.num_filters]
                    num_filters = args.num_filters
                else:
                    num_filters = [None]
                    suffix = [args.suffix]
                for i in range(len(num_filters)):
                    linear_probe(args.directory, blob, int(label_i),
                            suffix=suffix[i],
                            init_suffix=args.init_suffix,
                            num_filters=num_filters[i],
                            batch_size=args.batch_size,
                            ahead=args.ahead, quantile=args.quantile,
                            bias=args.bias, positive=args.positive, 
                            lower_bound=args.lower_bound, num_epochs=args.num_epochs, 
                            lr=args.learning_rate, l1_weight_decay=args.l1_decay,
                            l2_weight_decay=args.l2_decay,validation=args.validation,
                            min_train=args.min_train, max_train=args.max_train,
                            max_val=args.max_val,
                            cuda=cuda)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
