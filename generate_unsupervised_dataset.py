#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 00:51:40 2017

@author: dov84d
"""

#!/home/ohad/openface/venv/bin/python

import os
import os.path
import shutil
import errno
import glob
import math
import random
import itertools
import numpy as np
import scipy.io as sio
from scipy import ndimage
from scipy.misc import imsave
from scipy.misc import imresize


number_of_clusters = 6
patches_per_cluster_per_axis = 3 #without overlep
patch_size = [16, 16] #When changing patchs size need to change this
folder_name = 'patches2_16_16' #When changing patchs size need to change this
stride_size = patch_size[0]/4
distance_between_clusters = 3 #in patches units
max_tries = 1000
also_split_source_data_to_types = 1

def main():

    base_path = '/home/ron/Downloads/BSDS500-master/BSDS500/data'
    dataset_types = ['train', 'val', 'test']

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
                shutil.copyfile(img_filename, img_output_dir + "/" + base_filename + base_filename_ext)
                current_img = ndimage.imread(img_filename)
                GeneratePatches(patchs_output_dir, current_img, patch_size, base_filename)
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
                if (current_img.shape[0] > 500): # reduce image size
                   current_img = imresize(current_img, 50)
                   imsave(img_filename, current_img)
                GeneratePatches(patchs_output_dir, current_img, patch_size, base_filename)





def GeneratePatches(output_dir, img, patch_size, name_prefix):

    # we pad with zeros, so that patches can be obtained from all areas (including near the image boundary)
    pad_size = [(x + 1) / 2 for x in patch_size]
    padded_img = np.lib.pad(img, ((pad_size[0],), (pad_size[1],), (0,)), 'edge')

    cur_tries_num = 1
    dist_between_seeds = patches_per_cluster_per_axis/2.0 * patch_size[0] + distance_between_clusters*patch_size[0]
    clusters_seeds_list = []
    while len(clusters_seeds_list) < number_of_clusters and cur_tries_num < max_tries:
        cur_tries_num = cur_tries_num + 1
        i = np.random.randint(0, padded_img.shape[0] - patches_per_cluster_per_axis * patch_size[0])
        j = np.random.random_integers(0, padded_img.shape[1] - patches_per_cluster_per_axis * patch_size[1])
        too_close = 0
        for m,n in clusters_seeds_list:
            if math.sqrt((i - m)**2 + (n - j)**2) < dist_between_seeds:
                too_close = 1
                break

        if too_close:
            continue
        clusters_seeds_list.append([i,j])



    for cur_cluster in range(len(clusters_seeds_list)):
        dir_name = os.path.join(output_dir, name_prefix + '_' + format(cur_cluster, '02d'))
        makedirs_if_needed(dir_name)
        i = clusters_seeds_list[cur_cluster][0]
        j = clusters_seeds_list[cur_cluster][1]
        m = 0
        n = 0
        while n < patch_size[0] * (patches_per_cluster_per_axis - 1) + 1:
            while m < patch_size[1] * (patches_per_cluster_per_axis - 1) + 1:
              row_ind = i + n
              col_ind = j + m
              if row_ind+patch_size[0] < padded_img.shape[0] and col_ind+patch_size[1] < padded_img.shape[1]:
                current_patch = padded_img[row_ind:row_ind+patch_size[0], col_ind:col_ind+patch_size[1], :]
                filename_str = '{r:04d}_{c:04d}.png'.format(r=row_ind, c=col_ind)
                imsave(os.path.join(output_dir, name_prefix + '_' + format(cur_cluster, '02d'), filename_str), current_patch)
              m = m + stride_size
            m = 0
            n = n + stride_size




def makedirs_if_needed(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


if __name__ == "__main__":
    main()
