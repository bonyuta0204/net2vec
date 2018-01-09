import expdir

import torch
import torch.nn as nn
from torch.autograd import Variable
import loadseg
import time

import numpy as np


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


def upsample_blob_data(directory, blob, batch_size=64, verbose=True):
    ed = expdir.ExperimentDirectory(directory)
    if ed.has_mmap(blob=blob, part='upsampled'):
        print('%s already has %s, so skipping.' % (directory,
                                                   ed.mmap_filename(blob=blob, part='upsampled')))
    info = ed.load_info()
    blob_info = ed.load_info(blob=blob)
    shape = blob_info.shape
    N = shape[0]
    seg_size = get_seg_size(info.input_dim)

    blobdata = ed.open_mmap(blob=blob, mode='r', shape=shape)
    if verbose:
        print 'Creating new mmap at %s' % ed.mmap_filename(blob=blob, part='upsampled')
    upsampled_data = ed.open_mmap(blob=blob, part='upsampled', mode='w+',
                                  shape=(shape[0], shape[1], seg_size[0], seg_size[1]))

    up = nn.Upsample(size=seg_size, mode='bilinear')

    start_time = time.time()
    last_batch_time = start_time
    for i in range(int(np.ceil(np.true_divide(N, batch_size)))):
        if (i+1)*batch_size < N:
            idx = range(i*batch_size, (i+1)*batch_size)
        else:
            idx = range(i*batch_size, N)
        batch_time = time.time()
        rate = idx[-1] / (batch_time - start_time + 1e-15)
        batch_rate = batch_size / (batch_time - last_batch_time + 1e-15)
        last_batch_time = batch_time
        if verbose:
            print 'upsample_blob_data %d/%d (%.2f)\titems per sec %.2f\t%.2f' % (
                idx[-1], N, idx[-1]/float(N), batch_rate, rate)
        inp = Variable(torch.Tensor(blobdata[idx]))
        out = up(inp).data.cpu().numpy()
        upsampled_data[idx] = np.copy(out)

    if verbose:
        print 'Renaming mmap.'
    ed.finish_mmap(upsampled_data)


if __name__ == '__main__':
    import sys
    import traceback
    import argparse

    try:
        parser = argparse.ArgumentParser(description='TODO')
        parser.add_argument(
            '--directory',
            default='.',
            help='output directory for the net probe')
        parser.add_argument(
            '--blobs',
            nargs='*',
            help='network blob names to collect'
        )
        parser.add_argument(
            '--batch_size',
            type=int,
            default=64,
            help='TODO')

        args = parser.parse_args()
        for blob in args.blobs:
            upsample_blob_data(args.directory, blob, batch_size=args.batch_size, verbose=True)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
