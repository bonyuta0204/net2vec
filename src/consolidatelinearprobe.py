import expdir
import loadseg
import os
#from labelprobe import cached_memmap

def consolidate_probe(directory, blob, delete=False):
    ed = expdir.ExperimentDirectory(directory)
    info = ed.load_info()
    blob_info = ed.load_info(blob=blob)
    shape = blob_info.shape
    ds = loadseg.SegmentationData(info.dataset)

    L = ds.label_size() # number of labels (including background at index 0)
    N = ds.size() # total number of images in the dataset
    K = shape[1] # number of units for the given blob

    weights_mmap = ed.open_mmap(blob=blob, part='linear_weights', mode='w+',
            dtype='float32', shape=(L,K))

    missing_idx = []
    for l in range(L):
        if not ed.has_mmap(blob=blob, part='label_i_%d_weights' % l):
            missing_idx.append(l)
            continue
        try:
            label_weights_mmap = ed.open_mmap(blob=blob, part='label_i_%d_weights' % l,
                    mode='r', dtype='float32')
        except:
            # SUPPORT LEGACY CODE, TODO: remove eventually
            label_weights_mmap = ed.open_mmap(blob=blob, part='label_i_%d_weights' % l,
                    mode='r', dtype=float)
        weights_mmap[l] = label_weights_mmap[:]

    ed.finish_map(weights_mmap)
    print('Finished consolidating existing weights files ' 
            + '(Files for %d labels were missing).' % len(missing_idx))
    print(missing_idx)

    if delete:
        c = 0
        print('Deleting all unnecessary label weights files...')
        for l in range(L):
            if ed.has_mmap(blob=blob, part='label_i_%d_weights' % l):
                fn = ed.mmap_filename(blob=blob, part='label_i_%d_weights' % l)
                os.remove(fn)
                c += 1
        print('Removed %d weights files.' % c)


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparser.ArgumentParser()
        parser.add_argument('--directory', default='.')
        parser.add_argument('--blobs', nargs='*')
        parser.add_argument('--delete', action='store_true', default=False)

        args = parser.parse_args()

        for blob in args.blobs:
            consolidate_probe(args.directory, args.blobs, delete=args.delete)

    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
