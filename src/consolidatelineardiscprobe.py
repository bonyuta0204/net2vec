import expdir
import loadseg
import os
#from labelprobe import cached_memmap

def consolidate_disc_probe(directory, blob, bias=False, num_filters=None, suffix='', epochs=30, delete=False):
    ed = expdir.ExperimentDirectory(directory)
    info = ed.load_info()
    blob_info = ed.load_info(blob=blob)
    shape = blob_info.shape
    ds = loadseg.SegmentationData(info.dataset)

    L = ds.label_size() # number of labels (including background at index 0)
    N = ds.size() # total number of images in the dataset
    K = shape[1] # number of units for the given blob
    F = 1 if num_filters is None else len(num_filters) + 1

    if (ed.has_mmap(blob=blob, part='linear_weights_disc%s' % suffix)
        and (not bias or ed.has_mmap(blob=blob, part='linear_bias_disc%s' % suffix))):
        print('Linear weights (and bias) have already been consolidated')
    else:
        weights_mmap = ed.open_mmap(blob=blob, part='linear_weights_disc%s' % suffix, mode='w+',
                dtype='float32', shape=(L,F,K))
        
        if bias:
            bias_mmap = ed.open_mmap(blob=blob, part='linear_bias_disc%s' % suffix,
                    mode='w+', dtype='float32', shape=(L,F))

        results_mmap = ed.open_mmap(blob=blob, part='linear_results_disc%s' % suffix,
                                    mode='w+', dtype='float32', shape=(L, F, 4, epochs))

        if num_filters is None:
            suffixes = [suffix]
        else:
            suffixes = ['%s_num_filters_%d' % (suffix, n) for n in num_filters]
            suffixes.append(suffix)

        missing_idx = []
        for l in range(L):
            if not ed.has_mmap(blob=blob, part='label_i_%d_weights_disc%s' % (l, suffixes[0])):
                missing_idx.append(l)
                continue
            for i in range(len(suffixes)):
                suffix = suffixes[i]
                label_weights_mmap = ed.open_mmap(blob=blob, part='label_i_%d_weights_disc%s'
                        % (l, suffix), mode='r', dtype='float32', shape=(K,))

                weights_mmap[l,i,:] = label_weights_mmap[:]

                if bias:
                    label_bias_mmap = ed.open_mmap(blob=blob, part='label_i_%d_bias_disc%s'
                            % (l, suffix), mode='r', dtype='float32', shape=(1,))
                    bias_mmap[l,i] = label_bias_mmap[:]

                label_results_mmap = ed.open_mmap(blob=blob, part='label_i_%d_results_disc%s'
                                                  % (l, suffix), mode='r', dtype='float32',
                                                  shape=(4, epochs))
                results_mmap[l,i,:,:] = label_results_mmap[:,:]

        ed.finish_mmap(weights_mmap)
        if bias:
            ed.finish_mmap(bias_mmap)
        ed.finish_mmap(results_mmap)

        print('Finished consolidating existing weights files ' 
                + '(Files for %d labels were missing).' % len(missing_idx))
        print(missing_idx)

    if delete:
        c_w = 0
        c_b = 0
        c_r = 0
        print('Deleting all unnecessary label weights (and bias) files...')
        for l in range(L):
            for i in range(F):
                suffix = suffixes[i]
                if ed.has_mmap(blob=blob, part='label_i_%d_weights_disc%s' % (l, suffix)):
                    fn = ed.mmap_filename(blob=blob, part='label_i_%d_weights_disc%s' % (l, suffix))
                    os.remove(fn)
                    c_w += 1
                if bias and ed.has_mmap(blob=blob, part='label_i_%d_bias_disc%s' % (l, suffix)):
                    fn = ed.mmap_filename(blob=blob, part='label_i_%d_bias_disc%s' % (l, suffix))
                    os.remove(fn)
                    c_b += 1
                if ed.has_mmap(blob=blob, part='label_i_%d_results_disc%s' % (l, suffix)):
                    fn = ed.mmap_filename(blob=blob, part='label_i_%d_results_disc%s' % (l, suffix))
                    os.remove(fn)
                    c_r += 1

        print('Removed %d weights, %d bias, and %d results files.' % (c_w, c_b, c_r))


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--directory', default='.')
        parser.add_argument('--blobs', nargs='*')
        parser.add_argument('--bias', action='store_true', default=False)
        parser.add_argument('--suffix', type=str, default='')
        parser.add_argument('--num_filters', type=int, nargs='*', default=None)
        parser.add_argument('--epochs', type=int, default=30)
        parser.add_argument('--delete', action='store_true', default=False)

        args = parser.parse_args()

        for blob in args.blobs:
            consolidate_disc_probe(args.directory, blob, bias=args.bias,
                    num_filters=args.num_filters, epochs=args.epochs, suffix=args.suffix, delete=args.delete)

    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
