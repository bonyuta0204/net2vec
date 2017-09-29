import loadseg
import expdir

from indexdata import load_image_to_label
from labelprobe import cached_memmap
from labelprobe_pytorch import get_seg_size, iou_intersect_d, iou_union_d
from linearprobe_pytorch import CustomLayer

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import os
import time


def probe_linear(directory, blob, layer_i, batch_size=16, ahead=4, quantile=0.005, cuda=False):
    qcode = ('%f' % quantile).replace('0.','.').rstrip('0')
    ed = expdir.ExperimentDirectory(directory)
    if ed.has_mmap(blob=blob, part='label_i_%d_ious' % label_i):
        print('Label %d has already been probed, so skipping.' % label_i)
        return
    info = ed.load_info()
    seg_size = get_seg_size(info.input_dim)
    blob_info = ed.load_info(blob=blob)
    ds = loadseg.SegmentationData(info.dataset)
    shape = blob_info.shape
    N = shape[0] # number of total images
    K = shape[1] # number of units in given blob 
    L = ds.label_size() # number of labels
    if ed.has_mmap(blob=blob, part='label_i_%d_weights' % label_i):
        try:
            weights = ed.open_mmap(blob=blob, part='label_i_%d_weights' % label_i,
                    mode='r', dtype='float32', shape=(K,-1))
        except ValueError:
            # SUPPORTING LEGACY CODE (TODO: Remove)
            weights = ed.open_mmap(blob=blob, part='label_i_%d_weights' % label_i,
                    mode='r', dtype=float, shape=(K,-1))
    elif ed.has_mmap(blob=blob, parts='linear_weights'):
        all_weights = ed.open_mmap(blob=blob, part='linear_weights', mode='r',
                dtype='float32', shape=(L,K))
        weights = all_weights[label_i]
        if not np.any(weights):
            print('Label %d does not have associated weights to it, so skipping.') 
            return
    else:
        print('Label %d does not have associated weights to it, so skipping.')
        return

    quantdata = ed.open_mmap(blob=blob, part='quant-*', shape=(K, -1))
    threshold = quantdata[:, int(round(quantdata.shape[1] * quantile))]
    thresh = threshold[:, np.newaxis, np.newaxis]
    fn_read = ed.mmap_filename(blob=blob)
    label_categories = ds.label[label_i]['category'].keys()
    #num_categories = len(label_categories)
    label_name = ds.name(category=None, j=label_i)

    blobdata = cached_memmap(fn_read, mode='r', dtype='float32', shape=shape)
    image_to_label = load_image_to_label(directory)
    label_idx = np.where(image_to_label[:, label_i])[0]

    loader = loadseg.SegmentationPrefetcher(ds, categories=label_categories,
            indexes=label_idx, once=True, batch_size=batch_size, ahead=ahead,
            thread=True)
    num_imgs = len(loader.indexes)

    print('Probing with learned weights for label %d (%s) with %d images...' % (
        label_i, label_name, num_imgs)

    model = CustomLayer(K, upsample=True, up_size=seg_size, act=True, 
            positive=False)
    if cuda:
        model.cuda()

    iou_intersects = np.zeros(num_imgs)
    iou_unions = np.zeros(num_imgs)

    i = 0
    for batch in loader.batches():
        start_t = time.time()
        if (i+1)*batch_size < num_imgs:
            idx = range(i*batch_size, (i+1)*batch_size)
        else:
            idx = range(i*batch_size, num_imgs)
        i += 1
        input = torch.Tensor((blobdata[idx] > thresh).astype(float))
        input_var = (Variable(input.cuda(), volatile=True) if cuda else 
                Variable(input, volatile=True))

        target_ = []
        for rec in batch:
            for cat in label_categories:
                if rec[cat] != []:
                    if type(rec[cat]) is np.ndarray:
                        target_.append(np.max((rec[cat] == label_i).astype(float),
                            axis=0))
                    else:
                        target_.append(np.ones(seg_size)) 
                    break
        target = torch.Tensor(target_)
        target_var = (Variable(target.cuda(), volatile=True) if cuda
                else Variable(target, volatile=True))
        #target_var = Variable(target.unsqueeze(1).expand_as(
        #    input_var).cuda() if cuda
        #    else target.unsqueeze(1).expand_as(input_var))
        output_var = model(input_var)

        iou_intersects[idx] = np.squeeze(iou_intersect_d(output_var,
            target_var).data.cpu().numpy())
        iou_unions[idx] = np.squeeze(iou_union_d(output_var, 
            target_var).data.cpu().numpy())
        print('Batch: %d/%d\tTime: %f secs\tAvg Ind IOU: %f' % (i, 
            num_imgs/batch_size, 
            time.time()-start_t, np.mean(np.true_divide(iou_intersects[idx], 
                iou_unions[idx] + 1e-20))))
    
    loader.close()
    ind_ious = np.true_divide(iou_intersects, iou_unions + 1e-20)
    #print(ind_ious.shape)

    ind_ious_mmap = ed.open_mmap(blob=blob, part='label_i_%d_ious' % label_i, 
            mode='w+', dtype='float32', shape=(N,))
    ind_ious_mmap[label_idx] = ind_ious[:]
    ed.finish_mmap(ind_ious_mmap)


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--directory', default='.')
        parser.add_argument('--blobs', nargs='*')
        parser.add_argument('--labels', type=int, nargs='*')
        parser.add_argument('--start', type=int, default=None)
        parser.add_argument('--end', type=int, default=None)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--ahead', type=int, default=4)
        parser.add_argument('--quantile', type=float, default=0.005)
        parser.add_argument('--gpu', type=int, default=None)

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
        print(torch.cuda.device_count(), use_mult_gpu, cuda)
        for blob in args.blobs:
            for label_i in labels:
                probe_linear(args.directory, blob, label_i,
                        batch_size=args.batch_size, ahead=args.ahead,
                        quantile=args.quantile, cuda=cuda)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
