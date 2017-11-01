import loadseg
import expdir

from labelprobe import cached_memmap
from indexdata import has_image_to_label, load_image_to_label, create_image_to_label

import os
import time
import math

from customoptimizer_pytorch import Custom_SGD, Custom_Adam

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils import data

from torchvision.datasets.folder import default_loader
from torchvision import transforms

from torchnet.meter import AverageValueMeter

from matplotlib import pyplot as plt

import numpy as np


class CustomDiscLayer(nn.Module):
    def __init__(self, num_features, act=True, pool=None, positive=False, bias=False):
        super(CustomDiscLayer, self).__init__()
        self.num_features = num_features
        self.positive = positive
        self.weight = Parameter(torch.Tensor(self.num_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1))
        else:
            self.bias = 0
        assert (pool is 'avg_pool' or pool is 'max_pool' or pool is None)
        self.pool = pool
        self.act = act
        if act:
            self.activation = nn.Sigmoid()  # nn.LogSigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.num_features)
        if self.positive:
            # self.weight.data.uniform_(0, stdv/2)
            self.weight.data.uniform_(stdv / 2, stdv / 2)
        else:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        y = x * self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        if self.pool is 'max_pool':
            y = torch.max(torch.max(y, dim=3)[0], dim=2)[0]
        elif self.pool is 'avg_pool':
            y = torch.mean(torch.mean(y, dim=3), dim=2)
        if self.act:
            return self.activation((y.sum(1) + self.bias).squeeze())
        return (y.sum(1) + self.bias).squeeze()


class DiscriminativeData(data.Dataset):
    def __init__(self, segmentation, indexes, targets, split=None, transform=None,
                 target_transform=None, loader=default_loader):
        self.segmentation = segmentation
        self.split = split
        self.indexes = indexes
        self.imgs = []
        for i in range(len(self.indexes)):
            ind = self.indexes[i]
            if (split and segmentation.split(ind) == split) or not split:
                self.imgs.append((segmentation.filename(ind), targets[i]))
        if split:
            self.indexes = [i for i in self.indexes if segmentation.split(i) == split]
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def accuracy(output, target):
    return torch.div(torch.sum(torch.abs(target - torch.sigmoid(output)) < 0.5).type(torch.FloatTensor), output.size(0))
    # return torch.div(torch.sum(target * output > 0).type(torch.FloatTensor), output.size(0))


def run_epoch(activations, targets, act_idx, thresh, model, criterion, optimizer, epoch,
              batch_size=64, train=True, cuda=False):
    if train:
        model.train()
        volatile = False
    else:
        model.eval()
        volatile = True

    N = len(targets)
    assert (len(act_idx) == N)

    losses = AverageValueMeter()
    accuracies = AverageValueMeter()
    start = time.time()
    for i in range(int(np.ceil(N / float(batch_size)))):
        if (i + 1) * batch_size < N:
            idx = range(i * batch_size, (i + 1) * batch_size)
        else:
            idx = range(i * batch_size, N)

        # don't threshold is thresh is an array of all zeros
        if not any(thresh):
            input = torch.Tensor(activations[act_idx[idx]])
        else:
            input = torch.Tensor((activations[act_idx[idx]] > thresh).astype(float))
        target = torch.Tensor(targets[idx])
        input_var = Variable(input.cuda() if cuda else input, volatile=volatile)
        target_var = Variable(target.cuda() if cuda else target, volatile=volatile)
        output_var = model(input_var)
        loss = criterion(output_var, target_var)
        acc = accuracy(output_var, target_var)
        losses.add(loss.data[0])
        accuracies.add(acc.data[0])

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if train:
            #    print('Epoch {0}[{1}/{2}]\t'
            #          'Avg Loss {losses.avg:.4f}\t'
            #          'Overall IOU {3}\t'
            #          'Time {4}\t'.format(epoch, i, int(round(N/batch_size)),
            #              iou, time.time()-start, losses=losses))
            # else:
            #    print('Test [{0}/{1}]\t'
            #          'Avg Loss {losses.avg:.4f}\t'
            #          'Overall IOU {2}\t'
            #          'Time {3}\t'.format(i, int(round(N/batch_size)),
            #              iou, time.time()-start, losses=losses))

    if train:
        print('Epoch {0}\t'
              'Avg Loss {1}\t'
              'Acc {2}\t'
              'Time {3}\t'.format(epoch, losses.value()[0], accuracies.value()[0], time.time() - start))
    else:
        print('Test\t'
              'Avg Loss {0}\t'
              'Acc {1}\t'
              'Time {2}\t'.format(losses.value()[0], accuracies.value()[0], time.time() - start))

    return (losses.value()[0], accuracies.value()[0])


def adjust_learning_rate(starting_lr, optimizer, epoch, epoch_iter):
    lr = starting_lr * (0.1 ** (epoch // epoch_iter))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def linear_probe_discriminative(directory, blob, label_i, suffix='', batch_size=16,
                                quantile=0.005, bias=False, pool='avg_pool', positive=False,
                                num_epochs=30, epoch_iter=10, lr=1e-4, momentum=0.9,
                                l1_weight_decay=0, l2_weight_decay=0, nesterov=False,
                                lower_bound=None, optimizer_type='sgd', cuda=False,
                                fig_path=None, show_fig=True):
    # Make sure we have a directory to work in
    # qcode = ('%f' % quantile).replace('0.','').rstrip('0')
    ed = expdir.ExperimentDirectory(directory)
    # Check if linear weights have already been learned
    if ed.has_mmap(blob=blob, part='label_i_%d_weights_disc%s' % (label_i, suffix)):
        print('%s already has %s, so skipping.' % (directory,
                                                   ed.mmap_filename(blob=blob, part='label_i_%d_weights_disc%s' % (
                                                   label_i, suffix))))
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
        thresh = np.zeros((unit_size, 1, 1))
    else:
        quantdata = ed.open_mmap(blob=blob, part='quant-*', shape=(unit_size, -1))
        threshold = quantdata[:, int(round(quantdata.shape[1] * quantile))]
        thresh = threshold[:, np.newaxis, np.newaxis]
    # print np.max(thresh), thresh.shape, type(thresh)
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
        create_image_to_label(directory, batch_size=batch_size)
    image_to_label = load_image_to_label(directory)
    label_idx = np.where(image_to_label[:, label_i])[0]
    non_label_idx = np.where(image_to_label[:, label_i] == 0)[0]

    print('Number of positive and negative examples of label %d (%s): %d %d' % (
        label_i, label_name, len(label_idx), len(non_label_idx)))

    criterion = torch.nn.BCEWithLogitsLoss()
    layer = CustomDiscLayer(unit_size, act=False, pool=pool, bias=bias, positive=positive)
    if cuda:
        criterion.cuda()
        layer.cuda()

    if optimizer_type == 'sgd':
        optimizer = Custom_SGD(layer.parameters(), lr, momentum,
                               l1_weight_decay=l1_weight_decay, l2_weight_decay=l2_weight_decay,
                               nesterov=nesterov, lower_bound=lower_bound)
    elif optimizer_type == 'adam':
        optimizer = Custom_Adam(layer.parameters(), lr, l1_weight_decay=l1_weight_decay,
                                l2_weight_decay=l2_weight_decay, lower_bound=lower_bound)

    train_label_idx = []
    val_label_idx = []
    for ind in label_idx:
        if ds.split(ind) == 'train':
            train_label_idx.append(ind)
        elif ds.split(ind) == 'val':
            val_label_idx.append(ind)

    train_non_label_idx = []
    val_non_label_idx = []
    for ind in non_label_idx:
        if ds.split(ind) == 'train':
            train_non_label_idx.append(ind)
        elif ds.split(ind) == 'val':
            val_non_label_idx.append(ind)

    train_label_idx = np.array(train_label_idx)
    val_label_idx = np.array(val_label_idx)
    train_non_label_idx = np.array(train_non_label_idx)
    val_non_label_idx = np.array(val_non_label_idx)

    num_train_labels = len(train_label_idx)
    num_train_non_labels = len(train_non_label_idx)
    train_neg_greater = num_train_labels < num_train_non_labels

    if len(val_label_idx) < len(val_non_label_idx):
        pos_val_idx = np.arange(len(val_label_idx))
        neg_val_idx = np.random.choice(len(val_non_label_idx), len(val_label_idx), replace=False)
    else:
        pos_val_idx = np.random.choice(len(val_label_idx), len(val_non_label_idx), replace=False)
        neg_val_idx = np.arange(len(val_non_label_idx))

    val_indexes = np.concatenate((val_label_idx[pos_val_idx], val_non_label_idx[neg_val_idx]))
    val_targets = np.concatenate((np.ones(len(pos_val_idx)), np.zeros(len(neg_val_idx))))

    print 'Number of train and val examples: %d %d' % (min(2*num_train_non_labels, 2*num_train_labels),
                                                       len(val_targets))
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    for t in range(num_epochs):
        if epoch_iter is not None:
            adjust_learning_rate(lr, optimizer, t, epoch_iter=epoch_iter)
        if train_neg_greater:
            pos_train_idx = np.arange(num_train_labels)
            neg_train_idx = np.random.choice(num_train_non_labels, num_train_labels, replace=False)
        else:
            pos_train_idx = np.random.choice(num_train_labels, num_train_non_labels, replace=False)
            neg_train_idx = np.arange(num_train_non_labels)

        train_indexes = np.concatenate((train_label_idx[pos_train_idx], train_non_label_idx[neg_train_idx]))
        train_targets = np.concatenate((np.ones(len(pos_train_idx)), np.zeros(len(neg_train_idx))))
        rand_idx = np.random.permutation(len(train_targets))

        (train_loss, train_acc) = run_epoch(blobdata, train_targets[rand_idx], train_indexes[rand_idx], thresh, layer, criterion,
                      optimizer, t, batch_size=batch_size, train=True, cuda=cuda)
        (val_loss, val_acc) = run_epoch(blobdata, val_targets, val_indexes, thresh, layer, criterion,
                      optimizer, t, batch_size=batch_size, train=False, cuda=cuda)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    f, ax = plt.subplots(1,2, figsize=(8,4))
    ax[0].plot(range(num_epochs), train_losses, label='train')
    ax[0].plot(range(num_epochs), val_losses, label='val')
    ax[0].set_title('BCE Loss (train=%d, val=%d)' % (len(train_targets), len(val_targets)))
    ax[1].plot(range(num_epochs), train_accs, label='train')
    ax[1].plot(range(num_epochs), val_accs, label='val')
    ax[1].set_title('Accuracy (%s, %s)' % (label_name, blob))
    plt.legend()
    if fig_path is not None:
        plt.savefig(fig_path)
    if show_fig:
        plt.show()

    # Save weights
    weights = layer.weight.data.cpu().numpy()
    weights_mmap = ed.open_mmap(blob=blob, part='label_i_%d_weights_disc%s' % (label_i, suffix),
                                mode='w+', dtype='float32', shape=weights.shape)
    weights_mmap[:] = weights[:]
    ed.finish_mmap(weights_mmap)
    if bias:
        bias_v = layer.bias.data.cpu().numpy()
        bias_mmap = ed.open_mmap(blob=blob, part='label_i_%d_bias_disc%s' % (label_i, suffix),
                                 mode='w+', dtype='float32', shape=(1,))
        bias_mmap[:] = bias_v[:]
        ed.finish_mmap(bias_mmap)

    results_mmap = ed.open_mmap(blob=blob, part='label_i_%d_results_disc%s' % (label_i, suffix),
                                mode='w+', dtype='float32', shape=(4,num_epochs))
    results_mmap[0] = train_losses
    results_mmap[1] = val_losses
    results_mmap[2] = train_accs
    results_mmap[3] = val_accs
    ed.finish_mmap(results_mmap)


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
            default=64,
            help='the batch size to use')
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
            '--epoch_iter',
            type=int,
            default=None,
            help='the number of epochs to run before decreasing lr'
        )
        parser.add_argument(
            '--lower_bound',
            type=int,
            default=None,
            help='TODO')
        parser.add_argument(
            '--optimizer',
            type=str,
            default='sgd',
            help='Choose optimizer type from ["sgd", "adam"]'
        )
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
            '--figure_dir',
            type=str,
            default=None,
            help='TODO'
        )
        parser.add_argument(
            '--show_fig',
            action='store_true',
            default=False,
            help='TODO'
        )
        parser.add_argument(
            '--gpu',
            type=int,
            default=None,
            help='use GPU for training')

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
                if args.figure_dir is not None:
                    fig_path = os.path.join(args.figure_dir, blob, '%s%s.png' % (label_i, args.suffix))
                    directory = os.path.dirname(fig_path)
                    if not os.path.exists(directory):
                        print 'Creating %s ...' % directory
                        os.makedirs(directory)
                else:
                    fig_path = None

                linear_probe_discriminative(args.directory, blob, int(label_i),
                                            suffix=args.suffix,
                                            batch_size=args.batch_size,
                                            quantile=args.quantile,
                                            bias=args.bias,
                                            positive=args.positive,
                                            lower_bound=args.lower_bound,
                                            num_epochs=args.num_epochs,
                                            epoch_iter=args.epoch_iter,
                                            lr=args.learning_rate,
                                            optimizer_type=args.optimizer,
                                            cuda=cuda,
                                            fig_path=fig_path,
                                            show_fig=args.show_fig)

    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
