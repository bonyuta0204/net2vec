import os
import numpy
import re
import shutil
import time
import expdir
import sys
from scipy.io import savemat
import pickle as pkl

import loadseg

def max_probe(directory, blob, batch_size=None, quantile=0.005, results=None, num_components=None,
        suffix='', new_suffix='', normalize=True, pool='max_pool', should_thresh=False, disc=False):
    # Make sure we have a directory to work in
    ed = expdir.ExperimentDirectory(directory)

    # If it's already computed, then skip it!!!
    if disc:
        suffix = '_disc%s' % suffix
    print "Checking", ed.mmap_filename(blob=blob, part='linear_imgmax%s%s' % (suffix, new_suffix))
    if ed.has_mmap(blob=blob, part='linear_imgmax%s%s' % (suffix, new_suffix)):
        print "Already have %s-imgmax.mmap, skipping." % (blob)
        return

    info = ed.load_info()
    ds = loadseg.SegmentationData(info.dataset)

    # Read about the blob shape
    blob_info = ed.load_info(blob=blob)
    shape = blob_info.shape
    N = shape[0]
    K = shape[1]
    L  ds.label_size()

    if should_thresh:
        if quantile == 1:
            thresh = np.zeros((1,1,K,1,1))
        else:
            quantdata = ed.open_mmap(blob=blob, part='quant-*', shape=(K, -1))
            threshold = quantdata[:, int(round(quantdata.shape[1] * quantile))]
            thresh = threshold[numpy.newaxis,numpy.newaxis,:,numpy.newaxis,numpy.newaxis]
    
    print 'Computing imgmax for %s shape %r' % (blob, shape)
    data = ed.open_mmap(blob=blob, shape=shape)
    if results is not None:
        imgmax = ed.open_mmap(blob=blob, part='linear_imgmax%s%s' % (suffix, new_suffix),
                mode='w+', shape=(N,num_components))
        all_weights = pkl.load(open(results, 'rb'))['model'].x_weights_.T
    else:
        imgmax = ed.open_mmap(blob=blob, part='linear_imgmax%s%s' % (suffix, new_suffix),
                mode='w+', shape=(N,L))
        if disc:
            assert(ed.has_mmap(blob=blob, part='linear_weights%s' % suffix))
            all_weights = ed.open_mmap(blob=blob, part='linear_weights%s' % suffix,
                    mode='r', dtype='float32', shape=(L,2,K))
            all_weights = all_weights[:,-1,:]
        else:
            assert(ed.has_mmap(blob=blob, part='linear_weights%s' % suffix))
            all_weights = ed.open_mmap(blob=blob, part='linear_weights%s' % suffix,
                    mode='r', dtype='float32', shape=(L,K))
    if normalize:
        all_weights = numpy.array([numpy.true_divide(all_weights[i], numpy.linalg.norm(all_weights[i])) for i in range(L)])

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
        batch = data[i:i+batch_size][:,numpy.newaxis,:,:,:] # (batch_size, L, K, S, S)
        if should_thresh:
            batch = (batch > thresh).astype(float)
        if pool == 'max_pool':
            imgmax[i:i+batch_size,:] = (batch * all_weights[numpy.newaxis,:,:,numpy.newaxis,numpy.newaxis]).sum(axis=2).max(axis=(2,3))
        elif pool == 'avg_pool':
            imgmax[i:i+batch_size,:] = (batch * all_weights[numpy.newaxis,:,:,numpy.newaxis,numpy.newaxis]).sum(axis=2).mean(axis=(2,3))
    print 'Writing imgmax'
    sys.stdout.flush()
    # Save as mat file
    filename = ed.filename('linear_imgmax%s.mat' % suffix, blob=blob)
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
                '--results',
                type=str,
                default=None)
        parser.add_argument(
                '--num_components',
                type=int,
                default=None)
        parser.add_argument(
                '--disc',
                action='store_true',
                default=False)
        parser.add_argument(
                '--thresh',
                action='store_true',
                default=False)
        parser.add_argument(
                '--quantile',
                type=float,
                default=0.005)
        parser.add_argument(
                '--suffix',
                default='',
                type=str)
        parser.add_argument(
                '--new_suffix',
                default='',
                type=str)
        parser.add_argument(
                '--pool',
                default='max_pool',
                type=str)

        args = parser.parse_args()
        for blob in args.blobs:
            max_probe(args.directory, blob, args.batch_size, suffix=args.suffix,
                    normalize=args.normalize, results=args.results, num_components=args.num_components,
                    disc=args.disc, quantile=args.quantile,
                    should_thresh=args.thresh, new_suffix=args.new_suffix, 
                    pool=args.pool)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
