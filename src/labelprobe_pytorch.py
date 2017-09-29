import loadseg
import expdir

from labelprobe import cached_memmap
from indexdata import has_image_to_label, load_image_to_label, create_image_to_label

import os
import time

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable 


def iou_intersect_d(input, target, threshold = 0.5):
    return torch.sum(torch.sum(torch.mul((input > threshold).float(), target), dim=-1), 
                     dim=-1)


def iou_union_d(input, target, threshold = 0.5):
    return torch.sum(torch.sum(torch.clamp(torch.add(target, (input > threshold).float()), 
                                           max=1.), dim=-1), dim=-1)


def get_seg_size(input_dim):
    if input_dim == [227, 227]:
        seg_size = (113, 113)
    elif input_dim == [224, 224]:
        seg_size = (112, 112)
    elif input_dim == [384, 384]:
        seg_size = (192, 192)
    else:
        print input_dim
        assert(False)
    return seg_size


def label_probe(directory, blob, quantile=0.005, batch_size=16, ahead=4, start=None,
        end=None,cuda=False):
    # Make sure we have a directory to work in
    qcode = ('%f' % quantile).replace('0.','').rstrip('0')
    ed = expdir.ExperimentDirectory(directory)
    # Check if label probe has already been created
    if ed.has_mmap(blob=blob, part='single_iou'):
        print('%s already has %s, so skipping.' % (directory,
            ed.mmap_file(blob=blob, part='single_iou')))
    # Load probe metadata
    info = ed.load_info()
    seg_size = get_seg_size(info.input_dim)
    # Load blob metadata
    blob_info = ed.load_info(blob=blob)
    shape = blob_info.shape
    tot_imgs = shape[0]
    unit_size = shape[1]
    # Load the blob quantile data and grab thresholds
    quantdata = ed.open_mmap(blob=blob, part='quant-*', shape=(unit_size, -1))
    threshold = quantdata[:, int(round(quantdata.shape[1] * quantile))]
    thresh = threshold[:, np.newaxis, np.newaxis]
    # Map the blob activation data for reading
    fn_read = ed.mmap_filename(blob=blob)
    # Load the dataset
    ds = loadseg.SegmentationData(info.dataset)
    blobdata = cached_memmap(fn_read, mode='r', dtype='float32', shape=shape)
    # Get image-to-labels mapping
    if not has_image_to_label(directory):
        print('image_to_label does not exist in %s; creating it now...' % directory)
        create_image_to_label(directory, batch_size=batch_size, ahead=ahead)
    image_to_label = load_image_to_label(directory)

    num_labels = ds.label_size()
    upsample = nn.Upsample(size=seg_size, mode='bilinear')

    set_ious_mmap = ed.open_mmap(blob=blob, part='single_set_ious', mode='w+',
        dtype='float32', shape=(num_labels, unit_size))
    ind_ious_mmap = ed.open_mmap(blob=blob, part='single_ind_ious', mode='w+',
        dtype='float32', shape=(num_labels, tot_imgs, unit_size))
    
    if start is None:
        start = 1
    if end is None:
        end = num_labels
    #for label_i in range(1, num_labels):
    for label_i in range(start, end):
        print('Starting for label %d (%s)' % (label_i, ds.name(category=None,
            j=label_i)))
        label_categories = ds.label[label_i]['category'].keys()
        num_cats = len(label_categories)
        label_idx = np.where(image_to_label[:, label_i])[0]
        loader = loadseg.SegmentationPrefetcher(ds, categories=label_categories, 
                indexes=label_idx, once=False, batch_size=batch_size, 
                ahead=ahead, thread=True)
        loader_idx = loader.indexes
        N = len(loader_idx)
        iou_intersects = np.zeros((N, unit_size))
        iou_unions = np.zeros((N, unit_size)) 

        if num_cats > 1:
            rec_labcat = []
            for batch in loader.batches():
                for rec in batch:
                    for cat in label_categories:
                        if rec[cat] != []:
                            rec_labcat.append(cat)
                            break
        else:
            rec_labcat = [label_categories[0] for i in range(N)]


        i = 0
        for batch in loader.batches():
            start_t = time.time()
            if (i+1)*batch_size < N:
                idx = range(i*batch_size, (i+1)*batch_size)
            else:
                idx = range(i*batch_size, N)
            i += 1
            input = torch.Tensor((blobdata[loader_idx[idx]] > thresh).astype(float))
            input_var = upsample(Variable(input.cuda()) if cuda else
                    Variable(input))
            target = torch.Tensor([np.max((rec[rec_labcat[j]] 
                == label_i).astype(float), axis=0) 
                if type(rec[rec_labcat[j]]) is np.ndarray
                else np.ones(seg_size) for j, rec in enumerate(batch)])
            target_var = Variable(target.unsqueeze(1).expand_as(
                input_var).cuda() if cuda 
                else target.unsqueeze(1).expand_as(input_var))
            iou_intersects[idx] = np.squeeze(iou_intersect_d(input_var, 
                target_var).data.cpu().numpy())
            iou_unions[idx] = np.squeeze(iou_union_d(input_var, 
                target_var).data.cpu().numpy())
            print('Batch %d/%d\tTime %f secs' % (i, N/batch_size, time.time()-start_t))

        set_ious = np.true_divide(np.sum(iou_intersects, axis=0), 
                np.sum(iou_unions, axis=0) + 1e-20)
        loader.close()
        best_filter = np.argmax(set_ious)
        print('Label %d (%s): best set IOU = %f (filter %d)' % (label_i, 
            ds.name(category=None,j=label_i), set_ious[best_filter], best_filter))
        ind_ious = np.true_divide(iou_intersects, iou_unions + 1e-20)

        set_ious_mmap[label_i] = set_ious
        ind_ious_mmap[label_i, label_idx] = ind_ious

        #set_ious_mmap.flush()
        #ind_ious_mmap.flush()
    
    ed.finish_mmap(set_ious_mmap)
    ed.finish_mmap(ind_ious_mmap)


if __name__ == '__main__':
    import argparse
    import sys
    import traceback
    try:
        parser = argparse.ArgumentParser(
                description='Compute set IOUs for single filters')

        parser.add_argument('--directory', default='.', 
                help='output directory for the net probe')
        parser.add_argument('--blobs', nargs='*', 
                help='network blob names to tally')
        parser.add_argument('--quantile', type=float, default=0.005, 
                help='the quantile cutoff to use')
        parser.add_argument('--batch_size', default=16, type=int, 
                help='the batch size to use')
        parser.add_argument('--gpu', type=int, default=None,
                help='use GPU for training')
        parser.add_argument('--start', type=int, default=None)
        parser.add_argument('--end', type=int, default=None)

        args = parser.parse_args()
        
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
            label_probe(args.directory, blob, quantile=args.quantile, 
                    batch_size=args.batch_size, start=args.start, 
                    end=args.end, cuda=cuda)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
