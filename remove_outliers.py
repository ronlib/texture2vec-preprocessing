#!/usr/bin/python

import os
from collections import Counter
from PIL import Image
import glob
import numpy as np

MIN_FILES_IN_DIR = 8
MIN_SEGS = 3

#before 857,060 items, totalling 368.8 MB
#after
def main():
    base_path = "/media/dov84d/EHD2/TexturesProject/data/FiveKHalfAndBSRRemoveOutliers/patches2_32_32"
    dataset_types = ['train', 'val', 'test']
    for i in xrange(0,len(dataset_types)):
        dataset_type = dataset_types[i]
        data_dir = os.path.join(base_path, dataset_type)
        all_subdirs = next(os.walk(data_dir))[1]
        for cluster in all_subdirs:
            print("processing " + cluster)
            patches_list = []
            for patch in glob.glob(os.path.join(data_dir, cluster, '*.png')): #assuming gif
                im=Image.open(patch)
                patches_list.append(np.asarray(im))
            mean = np.mean(patches_list)
            std = np.std(patches_list)
            for patch in glob.glob(os.path.join(data_dir, cluster, '*.png')): #assuming gif
                im=Image.open(patch)
                cur_mean = np.mean(im)
                if np.abs(cur_mean - mean) > std/2:
                    print("Deleting " + patch)
                    os.remove(patch)

if __name__ == "__main__":
    main()