#!/usr/bin/env python
# coding: utf-8
import numpy as np
from patchMatch import patchMatch
from patchMatch import config
import cv2
from PIL import Image
from sep import constSep
from tqdm import tqdm
import time

if __name__ == "__main__":
    
    # get parameters
    opt = config() 
    assert opt.stride < opt.psize, 'stride should be smaller than psize'

    # setup images and paths
    ref_img = cv2.imread(opt.ref).astype(np.float)
    input_image = cv2.imread(opt.input).astype(np.float)
    hole_mask = opt.hole
    const_mask = opt.constraint

    # if constraint exists, using separater
    if const_mask != "None":
        # separate mask
        ## init separate model
        separater = constSep(hole_mask, const_mask)
        hole_mask = np.array(Image.open(hole_mask))
        hole_mask[np.where(hole_mask < 128)] = 0
        hole_mask[np.where(hole_mask >= 128)] = 255
        ## separate and get two masks
        left, right = separater.sep()
        masks = [left, right]
    # use hole mask directly otherwise
    else:
        # use hole mask directly
        hole_mask = np.array(Image.open(hole_mask))
        hole_mask[np.where(hole_mask < 128)] = 0
        hole_mask[np.where(hole_mask >= 128)] = 255
        masks = [hole_mask]

    # init patchMatch model
    model = patchMatch(ref_img, opt.psize)

    # set half size
    h_size = opt.psize // 2

    # iter through 2 sub-masks
    # record is used to save the patches
    record = []
    for index, mask in enumerate(masks):
        print("start mask num: {}".format(index))
        # find index of masked part
        places = np.where(mask == 0)

        # use original image directly
        input_img = input_image.copy()

        # set masked part to nan
        for i,j in zip(places[0], places[1]):
            input_img[i,j,:] = np.nan

        # check masked part
        test = np.nan_to_num(input_img.copy()).astype(np.uint8)
        cv2.imwrite("test_{}.jpg".format(index), test)

        # crop and search using nns
        print("start crop and match...")
        h,w,c = input_img.shape
        for i in tqdm(range(h_size, h-h_size, opt.stride)):
            for j in range(h_size, w-h_size, opt.stride):
                # if the pixel is in hole, crop and find
                if mask[i,j] == 0:
                    crop = input_img[i-h_size:i+h_size+1, j-h_size:j+h_size+1, :]
                    f = model.NNS(crop, opt.iteration, opt.workers)
                    # get small patch from patchMatch
                    tmp = model.reconstruction(f, crop)
                    # iter through the whole patch
                    ## if is masked, put pixel directly
                    ## otherwise, average the pixel values
                    for k in range(i-h_size, i+h_size+1):
                        for l in range(j-h_size, j+h_size+1):
                            if np.isnan(input_img[k,l,0]):
                                input_img[k,l,:] = tmp[k-i+h_size, l-j+h_size,:].copy()
                            else:
                                input_img[k,l,:] = (input_img[k,l,:] + tmp[k-i+h_size, l-j+h_size,:]) // 2
        # erode the mask to crop some neighbor part
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.erode(mask,kernel,iterations = 1)
        # record patches of masked part
        record.append(input_img * (1 - mask[:,:,None]/255))

    # mask input image again
    hole_mask = hole_mask // 255 
    input_image = input_image * hole_mask[:,:,None]
    
    # put patches on input image
    for i in range(h):
        for j in range(w):
            if const_mask != "None":
                if sum(record[0][i,j,:]) != 0 and sum(record[1][i,j,:]) != 0 and sum(input_image[i,j,:]) != 0:
                    input_image[i,j,:] = (record[0][i,j,:] + record[1][i,j,:] + input_image[i,j,:]) // 3
                elif sum(record[0][i,j,:]) == 0 and sum(record[1][i,j,:]) != 0 and sum(input_image[i,j,:]) != 0:
                    input_image[i,j,:] = (record[1][i,j,:] + input_image[i,j,:]) // 2
                elif sum(record[0][i,j,:]) != 0 and sum(record[1][i,j,:]) == 0 and sum(input_image[i,j,:]) != 0:
                    input_image[i,j,:] = (record[0][i,j,:] + input_image[i,j,:]) // 2
                elif sum(record[0][i,j,:]) != 0 and sum(record[1][i,j,:]) != 0 and sum(input_image[i,j,:]) == 0:
                    input_image[i,j,:] = (record[0][i,j,:] + record[1][i,j,:]) // 2
                elif sum(record[0][i,j,:]) == 0 and sum(record[1][i,j,:]) == 0 and sum(input_image[i,j,:]) != 0:
                    input_image[i,j,:] = input_image[i,j,:]
                elif sum(record[0][i,j,:]) == 0 and sum(record[1][i,j,:]) != 0 and sum(input_image[i,j,:]) == 0:
                    input_image[i,j,:] = record[1][i,j,:]
                elif sum(record[0][i,j,:]) != 0 and sum(record[1][i,j,:]) == 0 and sum(input_image[i,j,:]) == 0:
                    input_image[i,j,:] = record[0][i,j,:]
                else:
                    input_image[i,j,:] = 0
            else:
                if sum(record[0][i,j,:]) != 0 and sum(input_image[i,j,:]) != 0:
                    input_image[i,j,:] = (record[0][i,j,:] + input_image[i,j,:]) // 2
                elif sum(record[0][i,j,:]) == 0 and sum(input_image[i,j,:]) != 0:
                    input_image[i,j,:] = input_image[i,j,:]
                elif sum(record[0][i,j,:]) != 0 and sum(input_image[i,j,:]) == 0:
                    input_image[i,j,:] = record[0][i,j,:]
                else:
                    input_image[i,j,:] = 0

    print("finish!")
    cv2.imwrite(opt.output, input_image.astype(np.uint8))
    
