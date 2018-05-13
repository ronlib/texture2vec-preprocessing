#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 00:51:40 2017

@author: dov84d

This file use to create patchs from images that have segmentation estimation that was created by learning for next trasfer learning
"""

#!/home/ohad/openface/venv/bin/python

import os
import os.path
import shutil
import errno
import glob
import math
import argparse
import random
import itertools
import numpy as np
import scipy.io as sio
from scipy import ndimage
from scipy.misc import imsave
from scipy.misc import imresize


number_of_clusters = 6
patches_per_cluster_per_axis = 3 #without overlep
folder_name = 'patches_16_16' #When changing patchs size need to change this
distance_between_clusters = 3 #in patches units
max_tries = 1000
patches_per_image = 1000
also_split_source_data_to_types = 0

def main():

    parser = argparse.ArgumentParser(description='Semi supervised generator')
    parser.add_argument('-d', '--directory', required=True, dest='input_image_directory',
                        help='input image directory')
    parser.add_argument('-e', '--segments', required=True, dest='seg_est_path',
                        help='Segmentation estimates path')
    parser.add_argument('-s', '--size', required=True, dest='patch_length',
                        help='Patch length', default=16, type=int)
    parser.add_argument('-o', '--output-patches-directory', required=False,
                        dest='output_patches_directory',
                        help='Directory name of the output patches',
                        default="patches")

    args = parser.parse_args()

    base_path = args.input_image_directory
    dataset_types = ['train', 'val', 'test']
    # seg_est_path = '/home/ron/Downloads/BSDS500-master/BSDS500/data/groundTruth'
    seg_est_path = args.seg_est_path
    folder_name = args.output_patches_directory
    patch_size = [args.patch_length, args.patch_length]
    stride_size = patch_size[0]/4

    if also_split_source_data_to_types:

        source_img_path = os.path.join(base_path, 'images_source')
        target_img_path = os.path.join(base_path, 'images')
        img_list = glob.glob(os.path.join(source_img_path, '*.jpg'))
        random.shuffle(img_list)

        dataset_types_idx = [(0,int(0.8*len(img_list))), (int(0.8*len(img_list))+1,int(0.9*len(img_list))), (int(0.9*len(img_list))+1,len(img_list)-1)]

        for i in xrange(0,len(dataset_types)):
            img_output_dir = os.path.join(target_img_path, dataset_types[i])
            patchs_output_dir = os.path.join(base_path, folder_name, dataset_types[i])
            try:
                shutil.rmtree(img_output_dir)
            except OSError as exception:
                print(exception)
            try:
                shutil.rmtree(patchs_output_dir)
            except OSError as exception:
                print(exception)
            makedirs_if_needed(img_output_dir)
            makedirs_if_needed(patchs_output_dir)
            for j in xrange(dataset_types_idx[i][0],dataset_types_idx[i][1]):
                img_filename = img_list[j]
                print('Processing ' + img_filename)
                base_filename = os.path.splitext(os.path.basename(img_filename))[0]
                base_filename_ext = os.path.splitext(os.path.basename(img_filename))[1]
                current_img = ndimage.imread(img_filename)
                if (current_img.shape[0] > 500 or current_img.shape[1] > 500): # reduce image size
                   current_img = imresize(current_img, 50)
                imsave(img_output_dir + "/" + base_filename + base_filename_ext, current_img)

                #shutil.copyfile(img_filename, img_output_dir + "/" + base_filename + base_filename_ext)

                seg_est_filename = os.path.join(seg_est_path, base_filename + '.mat')
                if os.path.isfile(seg_est_filename):
                    d = sio.loadmat(seg_est_filename)
                    current_segmentation_est = d[0][0][0]
                    GeneratePatches(patchs_output_dir, current_img, current_segmentation_est, patch_size, base_filename)
    else:
        for i in xrange(0,len(dataset_types)):
            dataset_type = dataset_types[i]
            source_img_path = os.path.join(base_path, 'images', dataset_type)
            patchs_output_dir = os.path.join(base_path, folder_name, dataset_type)
            try:
                shutil.rmtree(patchs_output_dir)
            except OSError as exception:
                print(exception)
            makedirs_if_needed(patchs_output_dir)
            img_list = glob.glob(os.path.join(source_img_path, '*.jpg'))
            for img_filename in img_list:
                print('Processing ' + img_filename)
                base_filename = os.path.splitext(os.path.basename(img_filename))[0]
                base_filename_ext = os.path.splitext(os.path.basename(img_filename))[1]
                current_img = ndimage.imread(img_filename)

                #TODO reduce section not work yet
                #if (current_img.shape[0] > 500): # reduce image size
                   #current_img = imresize(current_img, 50)
                   #imsave(img_filename, current_img)
                seg_est_filename = os.path.join(seg_est_path, dataset_type, base_filename + '.mat')
                if os.path.isfile(seg_est_filename):
                    d = sio.loadmat(seg_est_filename)
                    current_segmentation_est = d['groundTruth'][0][0][0][0][0]
                    GeneratePatches(patchs_output_dir, current_img, current_segmentation_est, patch_size, base_filename)





def GeneratePatches(output_dir, img, img_segs_est, patch_size, name_prefix):

    # we pad with zeros, so that patches can be obtained from all areas (including near the image boundary)
    pad_size = [(x + 1) / 2 for x in patch_size]
    padded_img = np.lib.pad(img, ((pad_size[0],), (pad_size[1],), (0,)), 'edge')

    rows, cols = img.shape[0:2]
    patch_inds = random.sample(list(itertools.product(xrange(rows), xrange(cols))), patches_per_image)

    max_seg_label = np.amax(img_segs_est)
    for i in xrange(max_seg_label):
        dir_name = os.path.join(output_dir, name_prefix + '_' + format(i + 1, '02d'))
        makedirs_if_needed(dir_name)

    for row_ind, col_ind in patch_inds:
        current_seg = img_segs_est[row_ind, col_ind]
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
