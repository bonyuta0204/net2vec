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


def probe_linear(directory, blob, suffix='', start=None, end=None,batch_size=16, 
        ahead=4,  quantile=0.005, bias=False, positive=False, cuda=False, force=False):
    qcode = ('%f' % quantile).replace('0.','.').rstrip('0')
    ed = expdir.ExperimentDirectory(directory)
    if (ed.has_mmap(blob=blob, part='linear_ind_ious%s' % suffix) and 
            ed.has_mmap(blob=blob, part='linear_set_ious%s' % suffix)):
        print('Linear weights have already been probed.')
        print ed.mmap_filename(blob=blob, part='linear_set_val_ious%s' % suffix)
        if not force:
            return
        else:
            print('Forcefully continuing...')
    info = ed.load_info()
    seg_size = get_seg_size(info.input_dim)
    blob_info = ed.load_info(blob=blob)
    ds = loadseg.SegmentationData(info.dataset)
    shape = blob_info.shape
    N = shape[0] # number of total images
    K = shape[1] # number of units in given blob 
    L = ds.label_size() # number of labels

    if quantile == 1:
        thresh = np.zeros((K,1,1))
    else:
        quantdata = ed.open_mmap(blob=blob, part='quant-*', shape=(K, -1))
        threshold = quantdata[:, int(round(quantdata.shape[1] * quantile))]
        thresh = threshold[:, np.newaxis, np.newaxis]

    fn_read = ed.mmap_filename(blob=blob)
    blobdata = cached_memmap(fn_read, mode='r', dtype='float32', shape=shape)
    image_to_label = load_image_to_label(directory)

    if ed.has_mmap(blob=blob, part='linear_ind_ious%s' % suffix, inc=True) and not force:
        assert(ed.has_mmap(blob=blob, part='linear_set_ious%s' % suffix, inc=True))
        ind_ious = ed.open_mmap(blob=blob, part='linear_ind_ious%s' % suffix, mode='r+',
                inc=True, dtype='float32', shape=(L,N))
        set_ious = ed.open_mmap(blob=blob, part='linear_set_ious%s' % suffix, mode='r+',
                inc=True, dtype='float32', shape=(L,))
        set_ious_train = ed.open_mmap(blob=blob, part='linear_set_train_ious%s' % suffix,
                mode='r+', inc=True, dtype='float32', shape=(L,))
        try:
            set_ious_val = ed.open_mmap(blob=blob, part='linear_set_val_ious%s' % suffix, 
                    mode='r+', inc=True, dtype='float32', shape=(L,))
        except:
            set_ious_val = ed.open_mmap(blob=blob, part='linear_set_val_ious%s' % suffix, 
                    mode='r+',  dtype='float32', shape=(L,))


    else:
        ind_ious = ed.open_mmap(blob=blob, part='linear_ind_ious%s' % suffix, mode='w+',
                dtype='float32', shape=(L,N))
        set_ious = ed.open_mmap(blob=blob, part='linear_set_ious%s' % suffix, mode='w+',
                dtype='float32', shape=(L,))
        set_ious_train = ed.open_mmap(blob=blob, part='linear_set_train_ious%s' % suffix,
                mode='w+', dtype='float32', shape=(L,))
        set_ious_val = ed.open_mmap(blob=blob, part='linear_set_val_ious%s' % suffix, 
                mode='w+', dtype='float32', shape=(L,))

    if start is None:
        start = 1
    if end is None:
        end = L
    for label_i in range(start, end):
        if ed.has_mmap(blob=blob, part='label_i_%d_weights%s' % (label_i, suffix)):
            try:
                weights = ed.open_mmap(blob=blob, part='label_i_%d_weights%s' 
                        % (label_i, suffix),mode='r', dtype='float32', shape=(K,))
            except ValueError:
                # SUPPORTING LEGACY CODE (TODO: Remove)
                weights = ed.open_mmap(blob=blob, part='label_i_%d_weights%s' 
                        % (label_i, suffix), mode='r', dtype=float, shape=(K,))
        elif ed.has_mmap(blob=blob, part='linear_weights%s' % suffix):
            all_weights = ed.open_mmap(blob=blob, part='linear_weights%s' % suffix, 
                    mode='r', dtype='float32', shape=(L,K))
            weights = all_weights[label_i]
            if not np.any(weights):
                print('Label %d does not have associated weights to it, so skipping.' % label_i) 
                continue 
        else:
            print('Label %d does not have associated weights to it, so skipping.' % label_i)
            continue 

        if bias:
            if ed.has_mmap(blob=blob, part='label_i_%d_bias%s' % (label_i, suffix)):
                bias_v = ed.open_mmap(blob=blob, part='label_i_%d_bias%s' % 
                        (label_i, suffix), mode='r', dtype='float32', shape=(1,))
            else:
                assert(ed.has_mmap(blob=blob, part='linear_bias%s' % suffix))
                all_bias_v = ed.open_mmap(blob=blob, part='linear_bias%s' % suffix,
                        mode='r', dtype='float32', shape=(L,))
                bias_v = np.array([all_bias_v[label_i]])

        label_categories = ds.label[label_i]['category'].keys()
        label_name = ds.name(category=None, j=label_i)
        label_idx = np.where(image_to_label[:, label_i])[0]

        loader = loadseg.SegmentationPrefetcher(ds, categories=label_categories,
                indexes=label_idx, once=True, batch_size=batch_size, ahead=ahead,
                thread=True)
        loader_idx = loader.indexes
        num_imgs = len(loader.indexes)

        print('Probing with learned weights for label %d (%s) with %d images...' % (
            label_i, label_name, num_imgs))

        model = CustomLayer(K, upsample=True, up_size=seg_size, act=True, 
                bias=bias, positive=positive, cuda=cuda)
        model.weight.data[...] = torch.Tensor(weights)
        if bias:
            model.bias.data[...] = torch.Tensor(bias_v)

        if cuda:
            model.cuda()
        model.eval()

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
            input = torch.Tensor((blobdata[loader_idx[idx]] > thresh).astype(float))
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
        label_ind_ious = np.true_divide(iou_intersects, iou_unions + 1e-20)
        label_set_iou = np.true_divide(np.sum(iou_intersects), 
                np.sum(iou_unions) + 1e-20)

        ind_ious[label_i, loader_idx] = label_ind_ious
        set_ious[label_i] = label_set_iou
        train_idx = [i for i in range(len(loader_idx)) if ds.split(loader_idx[i]) == 'train']
        val_idx = [i for i in range(len(loader_idx)) if ds.split(loader_idx[i]) == 'val']
        set_ious_train[label_i] = np.true_divide(np.sum(iou_intersects[train_idx]),
                np.sum(iou_unions[train_idx]) + 1e-20)
        set_ious_val[label_i] = np.true_divide(np.sum(iou_intersects[val_idx]),
                np.sum(iou_unions[val_idx]) + 1e-20)

        print('Label %d (%s) Set IOU: %f, Train Set IOU: %f, Val Set IOU: %f, Max Ind IOU: %f' 
                % (label_i, label_name, label_set_iou, set_ious_train[label_i],
                    set_ious_val[label_i], np.max(label_ind_ious)))

    ed.finish_mmap(ind_ious)
    ed.finish_mmap(set_ious)
    ed.finish_mmap(set_ious_train)
    ed.finish_mmap(set_ious_val)


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--directory', default='.')
        parser.add_argument('--blobs', nargs='*')
        parser.add_argument('--suffix', type=str, default='')
        parser.add_argument('--start', type=int, default=1)
        parser.add_argument('--end', type=int, default=1198)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--ahead', type=int, default=4)
        parser.add_argument('--quantile', type=float, default=0.005)
        parser.add_argument('--bias', action='store_true', default=False)
        parser.add_argument('--positive', action='store_true', default=False)
        parser.add_argument('--force', action='store_true', default=False)
        parser.add_argument('--num_filters', type=int, nargs='*', default=None)
        parser.add_argument('--gpu', type=int, default=None)

        args = parser.parse_args()

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
            if args.num_filters is not None:
                suffixes = ['%s_num_filters_%d' % (args.suffix, n) for n in args.num_filters]
            else:
                suffixes = [args.suffix]
            for i in range(len(suffixes)):
                probe_linear(args.directory, blob, suffix=suffixes[i],
                        start=args.start, end=args.end, 
                        batch_size=args.batch_size, ahead=args.ahead,
                        quantile=args.quantile, bias=args.bias, 
                        positive=args.positive,
                        cuda=cuda,
                        force=args.force)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
