import expdir
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


def extract_concept_data(directory, batch_size=64, ahead=16, verbose=True):
    ed = expdir.ExperimentDirectory(directory)
    if ed.has_mmap(part='concept_data'):
        print('%s already has %s, so skipping' % (directory,
                                                  ed.mmap_filename(part='concept_data')))
        return
    info = ed.load_info()
    (sh, sw) = get_seg_size(info.input_dim)
    ds = loadseg.SegmentationData(info.dataset)
    categories = np.array(ds.category_names())
    L = ds.label_size()
    N = ds.size()
    pf = loadseg.SegmentationPrefetcher(ds, categories=categories, once=True, batch_size=batch_size,
                                        ahead=ahead, thread=True)

    if verbose:
        print 'Creating new mmap at %s' % ed.mmap_filename(part='concept_data')
    data = ed.open_mmap(part='concept_data', mode='w+', shape=(N,L,sh,sw))

    start_time = time.time()
    last_batch_time = start_time
    index = 0
    for batch in pf.batches():
        batch_time = time.time()
        rate = index / (batch_time - start_time + 1e-15)
        batch_rate = batch_size / (batch_time - last_batch_time + 1e-15)
        last_batch_time = batch_time
        if verbose:
            print 'extract_concept_data index %d/%d (%.2f)\titems per sec %.2f\t%.2f' % (index,
                                                                                        N,
                                                                                        index/float(N),
                                                                                        batch_rate,
                                                                                        rate)
        for rec in batch:
            for cat in categories:
                if len(rec[cat]) == 0:
                    continue
                if cat == 'texture' or cat == 'scene':
                    for i in range(len(rec[cat])):
                        data[index][rec[cat][i]-1,:,:] = 1
                else:
                    for i in range(len(rec[cat])):
                        ys, xs = np.where(rec[cat][i])
                        for j in range(len(xs)):
                            data[index][rec[cat][i][ys[j]][xs[j]]-1][ys[j]][xs[j]] = 1
            index += 1

    assert index == N, ("Data source should return every item once %d %d." %
                        (index, N))
    if verbose:
        print 'Renaming mmap.'
    ed.finish_mmap(data)


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
            '--batch_size',
            type=int,
            default=64,
            help='batch size')
        parser.add_argument(
            '--ahead',
            type=int,
            default=16,
            help='TODO')
        args = parser.parse_args()
        extract_concept_data(args.directory, batch_size=args.batch_size, ahead=args.ahead,
                             verbose=True)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)


