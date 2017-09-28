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
            act=True, positive=False):
        super(CustomLayer, self).__init__()
        self.num_features = num_features
        self.positive = positive
        self.weight = Parameter(torch.Tensor(self.num_features))
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
        y = x * self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        if self.act:
            if self.up:
                return self.activation(self.upsample(y.sum(1).unsqueeze(1)).squeeze())
            else:
                return self.activation(y.sum(1).squeeze())
        else:
            if self.up:
                return self.upsample(y.sum(1).unsqueeze(1)).squeeze()
            else:
                return y.sum(1).squeeze()


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


def run_epoch(activations, label_categories, label_i, fieldmap, thresh, sh, sw, 
        reduction, loader, model, criterion, optimizer, epoch, train=True, 
        cuda=False, iou_threshold=0.5):
    if train:
        model.train()
        volatile=False
    else:
        model.eval()
        volatile=True

    batch_size = loader.batch_size
    N = activations.shape[0]

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
        input = torch.Tensor((activations[idx] > thresh).astype(float))
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


def linear_probe(directory, blob, label_i, batch_size=16, ahead=4, 
        quantile=0.005, num_epochs=30, lr=1e-4, momentum=0.9, l1_weight_decay=0, 
        l2_weight_decay=0, nesterov=False, lower_bound=None, cuda=False):
    # Make sure we have a directory to work in
    qcode = ('%f' % quantile).replace('0.','').rstrip('0')
    ed = expdir.ExperimentDirectory(directory)
    # Check if linear weights have already been learned 
    if ed.has_mmap(blob=blob, part='label_i_%d_weights' % label_i):
        print('%s already has %s, so skipping.' % (directory,
            ed.mmap_filename(blob=blob, part='label_i_%d_weights' % label_i)))
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
    quantdata = ed.open_mmap(blob=blob, part='quant-*', shape=(unit_size, -1))
    threshold = quantdata[:, int(round(quantdata.shape[1] * quantile))]
    thresh = threshold[:, np.newaxis, np.newaxis]
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
    print('Total number of images containing label %d (%s): %d' % (
        label_i, label_name, len(label_idx)))
    
    try:
        train_loader = loadseg.SegmentationPrefetcher(ds, categories=label_categories, 
                split='train', indexes=label_idx, once=False, batch_size=batch_size, 
                ahead=ahead, thread=False)
    except IndexError as err:
        print(err.args)
        return
    
    train_idx = train_loader.indexes

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
    layer = CustomLayer(unit_size, upsample=True, up_size=(sh,sw), act=True, 
            positive=False)
    if cuda:
        layer.cuda()

    optimizer = Custom_SGD(layer.parameters(), lr, momentum,
            l1_weight_decay=l1_weight_decay, l2_weight_decay=l2_weight_decay,
            nesterov=nesterov, lower_bound=lower_bound)

    try:
        val_loader = loadseg.SegmentationPrefetcher(ds, categories=label_categories,
                split='val', indexes=label_idx, once=False, batch_size=batch_size,
                ahead=ahead, thread=True)
    except IndexError as err:
        print(err.args)
        train_loader.close()
        return

    val_idx = val_loader.indexes

    val_label_categories = []
    for batch in val_loader.batches():
        for rec in batch:
            for cat in label_categories:
                if rec[cat] != []:
                    val_label_categories.append(cat)
                    break
    assert(len(val_label_categories) == len(val_idx))

    for t in range(num_epochs):
        (_, iou) = run_epoch(blobdata[train_idx], train_label_categories, label_i,
                fieldmap, thresh, sh, sw, reduction, train_loader, layer, criterion, 
                optimizer, t+1, train=True, cuda=cuda, iou_threshold=0.5)
        (_, iou) = run_epoch(blobdata[val_idx], val_label_categories, label_i,
                fieldmap, thresh, sh, sw, reduction, val_loader, layer, criterion,
                optimizer, t+1, train=False, cuda=cuda, iou_threshold=0.5)

    # Close segmentation prefetcher (i.e. close pools)
    train_loader.close()
    val_loader.close()

    # Save weights
    weights = layer.weight.data.cpu().numpy()
    weights_mmap = ed.open_mmap(blob=blob, part='label_i_%d_weights' % label_i,
            mode='w+', dtype=float, shape=weights.shape)
    weights_mmap[:] = weights[:]
    ed.finish_mmap(weights_mmap)


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
                default=None,
                help='start index for class label (inclusive)')
        parser.add_argument(
                '--end',
                type=int,
                default=None,
                help='end index for class label (exclusive)')
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
                 '--gpu',
                type=int,
                default=None,
                help='use GPU for training')

        args = parser.parse_args()
        if args.start is not None and args.end is not None:
            labels = range(args.start, args.end)
        else:
            labels = args.labels

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
                linear_probe(args.directory, blob, int(label_i), 
                        batch_size=args.batch_size,
                        ahead=args.ahead, quantile=args.quantile,
                        num_epochs=args.num_epochs, lr=args.learning_rate, 
                        cuda=cuda)

    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
