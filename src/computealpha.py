import loadseg
import expdir
from indexdata import load_image_to_label, has_image_to_label, create_image_to_label

import numpy as np

def compute_alpha(directory):
    ed = expdir.ExperimentDirectory(directory)
    info = ed.load_info()
    ds = loadseg.SegmentationData(info.dataset)
    L = ds.label_size()
    if not has_image_to_label(directory):
        create_image_to_label(directory)
    image_to_label = load_image_to_label(directory)

    label_names = np.array([ds.label[i]['name'] for i in range(L)])

    alphas = np.zeros((L,))

    for label_i in range(1,L):
        label_categories = ds.label[label_i]['category'].keys()
        label_idx = np.where(image_to_label[:, label_i])[0]
        train_loader = loadseg.SegmentationPrefetcher(ds, categories=label_categories, split='train', indexes=label_idx,
                                                     once=True, batch_size=64, ahead=4, thread=True)
        train_idx = np.array(train_loader.indexes)
        #sw = 0
        #sh = 0
        perc_label = []
        #train_label_categories = []
        for batch in train_loader.batches():
            for rec in batch:
                sw, sh = [rec[k] for k in ['sw', 'sh']]
                #sw_r, sh_r = [rec[k] for k in ['sw', 'sh']]
                #if sw == 0 and sh == 0:
                #    sw = sw_r
                #    sh = sh_r
                #else:
                #    assert(sw == sw_r and sh == sh_r)
                for cat in label_categories:
                    if rec[cat] != []:
                        #train_label_categories.append(cat)
                        if type(rec[cat]) is np.ndarray:
                            perc_label.append(np.sum(rec[cat] == label_i) / float(sw * sh))
                        else:
                            perc_label.append(1.)
                        break
        assert(len(perc_label) == len(train_idx))

        alphas[label_i] = float(1. - np.mean(perc_label))
        print label_i, label_names[label_i], alphas[label_i]
        train_loader.close()

    alphas_mmap = ed.open_mmap(part='train_alphas', mode='w+', dtype='float32', shape=alphas.shape)
    alphas_mmap[:] = alphas[:]
    ed.finish_mmap(alphas_mmap)

if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser(
                description='Learn linear weights for maximizing segmentation loss\
                        for a given label')
        parser.add_argument(
                '--directory',
                default='.',
                help='output directory for the net probe')
        
        args = parser.parse_args()
        compute_alpha(args.directory)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)


