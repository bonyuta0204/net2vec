import numpy as np

import loadseg
import expdir


def has_image_to_label(directory):
    ed = expdir.ExperimentDirectory(directory)
    return ed.has_mmap(part='image_to_label')


def load_image_to_label(directory):
    ed = expdir.ExperimentDirectory(directory)                                  
    info = ed.load_info()                                                       
    ds = loadseg.SegmentationData(info.dataset)                                 
    shape = (ds.size(), len(ds.label))                                          
    return ed.open_mmap(part='image_to_label', mode='r', dtype=bool,
        shape=shape)


def create_image_to_label(directory, batch_size=16, ahead=4):
    ed = expdir.ExperimentDirectory(directory)
    info = ed.load_info()
    ds = loadseg.SegmentationData(info.dataset)
    categories = ds.category_names()
    shape = (ds.size(), len(ds.label))

    pf = loadseg.SegmentationPrefetcher(ds, categories=categories, once=True,
            batch_size=batch_size, ahead=ahead, thread=False)

    image_to_label = np.zeros(shape, dtype='int32')

    batch_count = 0
    for batch in pf.batches():
        if batch_count % 100 == 0:
            print('Processing batch %d ...' % batch_count)
        for rec in batch:
            image_index = rec['i']
            for cat in categories:
                if ((type(rec[cat]) is np.ndarray and rec[cat].size > 0) or
                        type(rec[cat]) is list and len(rec[cat]) > 0):
                    image_to_label[image_index][np.unique(rec[cat])] = True
        batch_count += 1

    mmap = ed.open_mmap(part='image_to_label', mode='w+', dtype=bool,
            shape=shape)
    mmap[:] = image_to_label[:]
    ed.finish_mmap(mmap)
    f = ed.mmap_filename(part='image_to_label')

    print('Finished and saved index_to_label at %s' % f)


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser(
                description='Generate an image-to-label array that maps \
                        each image in the dataset to the labels included in it.')
        parser.add_argument('--directory', help='experiment directory for net probe')
        parser.add_argument('--ahead', type=int, default=4, 
                help='the prefetch lookahead size')
        parser.add_argument('--batch_size', type=int, default=16, 
                help='the batch size to use')
        args = parser.parse_args()
        create_image_to_label(args.directory, batch_size=args.batch_size,
                ahead=args.ahead)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
