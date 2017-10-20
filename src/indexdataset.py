import numpy as np

import loadseg
import expdir


DATASETS = ['pascal', 'ade20k', 'opensurfaces', 'dtd']


def get_dataset_index(fn):
    for i in range(len(DATASETS)):
        if DATASETS[i] in fn:
            return i


def create_image_dataset_label_index(directory, batch_size=64, ahead=16):
    ed = expdir.ExperimentDirectory(directory)
    info = ed.load_info()
    ds = loadseg.SegmentationData(info.dataset)
    categories = ds.category_names()
    shape = (ds.size(), len(DATASETS), len(ds.label))
    index = np.zeros(shape)
    pf = loadseg.SegmentationPrefetcher(ds, categories=categories,
                                        once=True, batch_size=batch_size, 
                                        ahead=ahead, thread=True)
    batch_count = 0
    for batch in pf.batches():
        if batch_count % 100 == 0:
            print('Processing batch %d ...' % batch_count)
        for rec in batch:
            dataset_index = get_dataset_index(rec['fn'])
            image_index = rec['i']
            for cat in categories:
                if ((type(rec[cat]) is np.ndarray and rec[cat].size > 0) or     
                        type(rec[cat]) is list and len(rec[cat]) > 0):       
                    index[image_index][dataset_index][np.unique(rec[cat])] = True
        batch_count += 1

    mmap = ed.open_mmap(part='image_dataset_label', mode='w+', dtype=bool,
            shape=shape)
    mmap[:] = index[:]
    ed.finish_mmap(mmap)
    print('Finished and saved at %s' % ed.mmap_filename(part='image_dataset_label'))


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:                                                                        
        parser = argparse.ArgumentParser(                                       
                description='Generate an image-dataset-label matrix index')
        parser.add_argument('--directory', help='experiment directory for net probe')
        parser.add_argument('--ahead', type=int, default=16,                     
                help='the prefetch lookahead size')                             
        parser.add_argument('--batch_size', type=int, default=64,               
                help='the batch size to use')                                   
        args = parser.parse_args()                                              
        create_image_dataset_label_index(args.directory, batch_size=args.batch_size,       
                ahead=args.ahead)                                               
    except:                                                                     
        traceback.print_exc(file=sys.stdout)                                    
        sys.exit(1)                                                             

