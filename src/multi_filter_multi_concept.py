import expdir
import loadseg
import time

import numpy as np
import pickle as pkl

from sklearn.cross_decomposition import CCA
from upsample_blob_data import get_seg_size


def compute_correlation(directory, blob, num_samples=None, num_components=1, out_file=None,
        verbose=False):
    ed = expdir.ExperimentDirectory(directory)

    info = ed.load_info()
    ds = loadseg.SegmentationData(info.dataset)
    
    L = ds.label_size()
    N = ds.size()

    blob_info = ed.load_info(blob=blob)
    shape = blob_info.shape
    K = shape[1]

    categories = np.array(ds.category_names())
    label_names = np.array([ds.label[i]['name'] for i in range(L)])

    (Hs, Ws) = get_seg_size(info.input_dim)

    if verbose:
        start = time.time()
        print 'Loading data...'
    upsampled_data = ed.open_mmap(blob=blob, part='upsampled', mode='r',
            shape=(N,K,Hs,Ws))
    concept_data = ed.open_mmap(part='concept_data', mode='r',
            shape=(N,L,Hs,Ws))
    if verbose:
        print 'Finished loading data in %d secs.' % (time.time() - start)

    if verbose:
        start = time.time()
        print 'Selecting data...'

    if num_samples is not None:
        rand_idx = np.random.choice(N, num_samples, replace=False)
        X = upsampled_data[rand_idx,:,Hs/2,Ws/2]
        Y = concept_data[rand_idx,:,Hs/2,Ws/2]
    else:
        X = upsampled_data[:,:,Hs/2,Ws/2]
        Y = concept_data[:,:,Hs/2,Ws/2]

    if verbose:
        print 'Finished selecting data in %d secs.' % (time.time() - start)

    cca = CCA(n_components=num_components)

    if verbose:
        start = time.time()
        if num_samples is None:
            num_samples = N
        print 'Fitting %d-component CCA with N = %d samples...' % (num_components, num_samples)
    cca.fit(X,Y)
    if verbose:
        print 'Fitted %d-component CCA with N = %d samples in %d secs.' % (num_components,
                num_samples, time.time() - start)

    X_c, Y_c = cca.transform(X,Y)
    score = cca.score(X,Y)

    results = {}
    if out_file is not None:
        if verbose:
            start = time.time()
            print 'Saving results...'
        results['model'] = cca
        try:
            results['idx'] = rand_idx
        except:
            results['idx'] = None 
        results['directory'] = directory
        results['blob'] = blob
        results['num_samples'] = num_samples
        results['num_components'] = num_components
        results['score'] = score

        pkl.dump(results, open(out_file, 'wb'))
        if verbose:
            print 'Saved results at %s in %d secs.' % (out_file, time.time() - start)

    return results


if __name__ == '__main__':
    import sys
    import argparse
    import traceback

    try:
        parser = argparse.ArgumentParser(description='TODO')
        parser.add_argument('--directory',
                help='TODO')
        parser.add_argument('--blob',
                type=str,
                help='TODO')
        parser.add_argument('--num_samples',
                type=int,
                default=None,
                help='TODO')
        parser.add_argument('--num_components',
                type=int,
                default=1,
                help='TODO')
        parser.add_argument('--out_file',
                type=str,
                default=None,
                help='TODO')

        args = parser.parse_args()
        compute_correlation(args.directory, args.blob, num_samples=args.num_samples, 
                num_components=args.num_components, out_file=args.out_file, verbose=True)
    except:
       traceback.print_exc(file=sys.stdout)
       sys.exit(1)
