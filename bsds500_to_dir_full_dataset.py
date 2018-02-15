#!/home/ohad/openface/venv/bin/python

import os
import os.path
import errno
import glob
import random
import itertools
import numpy as np
import scipy.io as sio
from scipy import ndimage
from scipy.misc import imsave


def main():
#    patch_size = [64, 64]
    patch_size = [32, 32]
#    patch_size = [16, 16]

    patches_per_image = 10000

    base_path = '/home/ron/Downloads/BSDS500-master/BSDS500/data'
    dataset_type = 'train'

    gt_path = os.path.join(base_path, 'groundTruth', dataset_type)
    img_path = os.path.join(base_path, 'images', dataset_type)

    output_dir = os.path.join("/home/ron/Downloads/BSDS500-master/patches", dataset_type)
    #output_dir = os.path.join("/home/ohad/data/patches2_16_16/", dataset_type)

    img_list = glob.glob(os.path.join(img_path, '*.jpg'))

    for img_filename in img_list:
        print('Processing ' + img_filename)

        base_filename = os.path.splitext(os.path.basename(img_filename))[0]
        gt_filename = os.path.join(gt_path, base_filename + '.mat')

        d = sio.loadmat(gt_filename)
        current_segmentation = d['groundTruth'][0, 0]['Segmentation'][0, 0]
        current_img = ndimage.imread(img_filename)

        rows, cols = current_img.shape[0:2]
        # Generate pairs of random inds (without replacement)
        all_inds = random.sample(list(itertools.product(xrange(rows), xrange(cols))), patches_per_image)

        GeneratePatches(output_dir, all_inds, current_img, current_segmentation, patch_size, base_filename)


def GeneratePatches(output_dir, patch_inds, img, img_segs, patch_size, name_prefix):
    # we pad with zeros, so that patches can be obtained from all areas (including near the image boundary)
    pad_size = [(x + 1) / 2 for x in patch_size]
    padded_img = np.lib.pad(img, ((pad_size[0],), (pad_size[1],), (0,)), 'edge')

    # segment labels are between 1 and max_seg_label (zero indicates an invalid region -- the pad area)
    max_seg_label = np.amax(img_segs)
    for i in xrange(max_seg_label):
        dir_name = os.path.join(output_dir, name_prefix + '_' + format(i + 1, '02d'))
        makedirs_if_needed(dir_name)

    for row_ind, col_ind in patch_inds:
        current_seg = img_segs[row_ind, col_ind]
        current_patch = padded_img[row_ind:row_ind+patch_size[0]-1, col_ind:col_ind+patch_size[1]-1, :]

        filename_str = '{r:04d}_{c:04d}.png'.format(r=row_ind, c=col_ind)
        imsave(os.path.join(output_dir, name_prefix + '_' + format(current_seg, '02d'), filename_str), current_patch)


def makedirs_if_needed(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


if __name__ == "__main__":
    main()
