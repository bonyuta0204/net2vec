import os
import numpy
import re
import shutil
import time
import expdir
import sys
from scipy.io import savemat

import loadseg

def max_probe(directory, blob, batch_size=None, quantile=0.005, 
        suffix='', normalize=True):
    # Make sure we have a directory to work in
    ed = expdir.ExperimentDirectory(directory)

    # If it's already computed, then skip it!!!
    print "Checking", ed.mmap_filename(blob=blob, part='imgmax')
    if ed.has_mmap(blob=blob, part='imgmax'):
        print "Already have %s-imgmax.mmap, skipping." % (blob)
        return

    info = ed.load_info()
    ds = loadseg.SegmentationData(info.dataset)

    # Read about the blob shape
    blob_info = ed.load_info(blob=blob)
    shape = blob_info.shape
    N = shape[0]
    K = shape[1]
    L = ds.label_size()

    #if quantile == 1:
    #    thresh = np.zeros((K,1,1))
    #else:
    #    quantdata = ed.open_mmap(blob=blob, part='quant-*', shape=(K, -1))
    #    threshold = quantdata[:, int(round(quantdata.shape[1] * quantile))]
    #    thresh = threshold[:, np.newaxis, np.newaxis]
    
    print 'Computing imgmax for %s shape %r' % (blob, shape)
    data = ed.open_mmap(blob=blob, shape=shape)
    imgmax = ed.open_mmap(blob=blob, part='linear_imgmax',
            mode='w+', shape=(N,L,K))

    assert(ed.has_mmap(blob=blob, part='linear_weights%s' % suffix))
    all_weights = ed.open_mmap(blob=blob, part='linear_weights%s' % suffix,
            mode='r', dtype='float32', shape=(L,K))
    if normalize:
        all_weights = np.true_divide(all_weights, np.sum(all_weights, axis=1))

    # Automatic batch size selection: 64mb batches
    if batch_size is None:
        batch_size = max(1, int(128 * 1024 * 1024 / numpy.prod(shape[1:])))

    # Algorithm: one pass over the data
    start_time = time.time()
    last_batch_time = start_time
    for i in range(0, data.shape[0], batch_size):
        batch_time = time.time()
        rate = i / (batch_time - start_time + 1e-15)
        batch_rate = batch_size / (batch_time - last_batch_time + 1e-15)
        last_batch_time = batch_time
        print 'Imgmax %s index %d: %f %f' % (blob, i, rate, batch_rate)
        sys.stdout.flush()
        batch = data[i:i+batch_size][:,np.newaxis,:,:,:]
        imgmax[i:i+batch_size,:] = (batch * all_weights[np.newaxis,:,:]).sum(axis=2).max(axis=(2,3))
    print 'Writing imgmax'
    sys.stdout.flush()
    # Save as mat file
    filename = ed.filename('linear_imgmax.mat', blob=blob)
    savemat(filename, { 'linear_imgmax': imgmax })
    # And as mmap
    ed.finish_mmap(imgmax)


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        import loadseg

        parser = argparse.ArgumentParser(
            description='Generate sorted files for probed activation data.')
        parser.add_argument(
                '--directory',
                default='.',
                help='output directory for the net probe')
        parser.add_argument(
                '--blobs',
                nargs='*',
                help='network blob names to sort')
        parser.add_argument(
                '--batch_size',
                type=int, default=None,
                help='the batch size to use')
        parser.add_argument(
                '--normalize',
                action='store_true',
                default=False)
        parser.add_argument(
                '--suffix',
                default='',
                type=str)

        args = parser.parse_args()
        for blob in args.blobs:
            max_probe(args.directory, blob, args.batch_size, suffix=args.suffix,
                    normalize=args.normalize)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
