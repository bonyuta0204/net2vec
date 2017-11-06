import expdir
import loadseg
from indexdata import load_image_to_label

import numpy as np

def save_neural_statistics(directory, blob, split=None):
    ed = expdir.ExperimentDirectory(directory)
    info = ed.load_info()
    dataset = info.dataset
    blob_info = ed.load_info(blob=blob)

    acts = ed.open_mmap(blob=blob, mode='r', dtype='float32', shape=blob_info.shape)

    if 'broden' in dataset:
        assert(split is not None)
        ds = loadseg.SegmentationData(dataset)
        split_ind = np.array([True if ds.split(i) == split else False for i in range(ds.size())])
        L = ds.label_size()
    elif 'imagenet' in dataset or 'ILSVRC' in dataset:
        L = 1000
        split_ind = np.array(L*[True])
    image_to_label = load_image_to_label(directory)

    K = blob_info.shape[1]

    probs = np.zeros((L,K))
    mus = np.zeros((L,K))
    sigmas = np.zeros(L,K)

    for class_i in range(L):
        class_idx = np.where(split_ind & image_to_label[:,class_i])[0]
        max_acts = np.amax(acts[class_idx], axis=(2,3))
        for filter_i in range(blob_info.shape[1]):
            nz_idx = np.where(max_acts[:,filter_i] > 0)[0]
            probs[class_i][filter_i] = len(nz_idx)/float(len(max_acts[:,filter_i]))
            mus[class_i][filter_i] = np.mean(max_acts[nz_idx,filter_i])
            sigmas[class_i][filter_i] = np.std(max_acts[nz_idx,filter_i])

    if 'broden' in dataset:
        suffix = '_%s' % split
    else:
        suffix = ''

    probs_mmap = ed.open_mmap(blob=blob, part='act_probs%s' % suffix, mode='w+', dtype='float32',
                              shape=probs.shape)
    mus_mmap = ed.open_mmap(blob=blob, part='act_mus%s' % suffix, mode='w+', dtype='float32',
                            shape=mus.shape)
    sigmas_mmap = ed.open_mmap(blob=blob, part='act_sigmas%s' % suffix, mode='w+', dtype='float32',
                               shape=sigmas.shape)

    probs_mmap[:] = probs[:]
    mus_mmap[:] = mus[:]
    sigmas_mmap[:] = sigmas[:]

    ed.finish_mmap(probs_mmap)
    ed.finish_mmap(mus_mmap)
    ed.finish_mmap(sigmas_mmap)


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--directory', default='.')
        parser.add_argument('--blobs', nargs='*')
        parser.add_argument('--split', type=str, default=None)

        args = parser.parse_args()

        for blob in args.blobs:
            save_neural_statistics(args.directory, blob, args.split)

    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
