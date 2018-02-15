#!/usr/bin/python

import os
import shutil
from collections import Counter

MIN_FILES_IN_DIR = 8
MIN_SEGS = 3


def remove_sparse_dirs(data_dir):
    all_subdirs = next(os.walk(data_dir))[1]

    for x in all_subdirs:
        if len(next(os.walk(os.path.join(data_dir, x)))[2]) < MIN_FILES_IN_DIR:
            print("Deleting " + x)
            shutil.rmtree(os.path.join(data_dir, x))


def remove_too_few_segs(data_dir):
    all_subdirs = next(os.walk(data_dir))[1]
    seg_count_per_file = Counter([x.rsplit('_', 1)[0] for x in all_subdirs])

    for x in all_subdirs:
        x_basename = x.rsplit('_', 1)[0]
        if seg_count_per_file[x_basename] < MIN_SEGS:
            print("Deleting " + x)
            #print("Because {}".format(seg_count_per_file[x_basename]))
            shutil.rmtree(os.path.join(data_dir, x))

    print(seg_count_per_file)


def main():

    dataset_types = ['train', 'val', 'test']
    for i in xrange(0,len(dataset_types)):
        base_data_dir = "/home/ron/Downloads/BSDS500-master/BSDS500/data/patches2_32_32"
        data_dir = os.path.join(base_data_dir, dataset_types[i])

        # Remove directories with few files
        remove_sparse_dirs(data_dir)

        # Remove directories where there aren't enough segments
        remove_too_few_segs(data_dir)


if __name__ == "__main__":
    main()
