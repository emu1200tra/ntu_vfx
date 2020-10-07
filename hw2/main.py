from __future__ import print_function
import cv2
import numpy as np
import argparse
from multiprocessing import Process, Pool
from util import *
from tqdm import tqdm
import os
import time

def parse():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", help="path for dir", required=True, type=str)
  parser.add_argument("--txt", help="path for txt", required=True, type=str)
  parser.add_argument("--out", help="path for output", default='./new_result.jpg', type=str)
  parser.add_argument("--plot", help="plot or not", default=False, type=bool)
  parser.add_argument("--showP", help="show feature point", default=False, type=bool)
  args = parser.parse_args()
  return args

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def main():
  total_time = 0.0
  args = parse()
  print ('Reading images...')
  t = time.time()
  images, focal = read_img(args.dir, args.txt)
  images = prepare_image(images, focal)#[::-1]
  print('image shape:', images[0].shape)
  print('Reading images done in ', time.time()-t)
  total_time += time.time()-t
  images_gray = []
  os.system("mkdir cyl_input_images")
  os.system("mkdir gray_input_images")
  print('Write out prepared input images...')
  t = time.time()
  for i in range(len(images)):
    cv2.imwrite('cyl_input_images/'+str(i)+'.jpg', images[i])
    if args.plot:
      cv2.imwrite('tmp.jpg', images[i])
      jizz = cv2.imread('tmp.jpg')
      cv2.imshow('jizz', jizz)
      cv2.waitKey(0)
    Y = rgb2gray(images[i])
    cv2.imwrite('gray_input_images/'+str(i)+'.jpg', Y)
    if args.plot:
      cv2.imwrite('tmp.jpg', Y)
      jizz = cv2.imread('tmp.jpg')
      cv2.imshow('jizz', jizz)
      cv2.waitKey(0)
    images_gray.append(cv2.GaussianBlur(Y, (5,5), 0))
  print('Write images done in ', time.time()-t)
  total_time += time.time()-t

  print('Processing Harris...')
  t = time.time()
  p = Pool(processes = len(images_gray))
  data = p.map(harris, [i for i in images_gray])
  p.close()
  
  pos = []
  desc = []

  for i in data:
    pos1, desc1 = i  
    pos.append(pos1)
    desc.append(desc1)
  print('Harris done in ', time.time()-t)
  total_time += time.time()-t
  font = cv2.FONT_HERSHEY_SIMPLEX
  fontScale = 1
  fontColor = (0,0,255)
  lineType = 2
  if args.showP:
    for j in range(2):
      text = np.array(images[j])
      for i in pos[j]:
        cv2.putText(text, 'x', (i[0], i[1]), font, fontScale, fontColor, lineType)
      cv2.imwrite('feature_point_'+str(j)+'.jpg', text)
      text = cv2.imread('feature_point_'+str(j)+'.jpg')
      if args.plot:
        cv2.imshow('feature', text)

  print('Processing feature matching...')
  t = time.time()
  p = Pool(processes = len(pos)-1)
  data = p.map(feature_cmp, [(desc[i], desc[i+1], pos[i], pos[i+1], 0.8) for i in range(len(pos)-1)])
  p.close()

  goods = []

  for i in data:
    goods.append(i)
  print('Feature matching done in ', time.time()-t)

  if args.showP:
    text1 = np.array(images[0])
    text2 = np.array(images[1])
    p1 = [i[1] for i in goods[0]]
    p0 = [i[2] for i in goods[0]]
    for i,j in zip(p1, p0):
      cv2.putText(text1, 'x', (i[0], i[1]), font, fontScale, fontColor, lineType)
      cv2.putText(text2, 'x', (j[0], j[1]), font, fontScale, fontColor, lineType)
    cv2.imwrite('matching_0.jpg', text1)
    text1 = cv2.imread('matching_0.jpg')
    cv2.imwrite('matching_1.jpg', text2)
    text2 = cv2.imread('matching_1.jpg')
    if args.plot:
      cv2.imshow('matching', text1)
      cv2.imshow('matching', text2)

  total_time += time.time()-t
  t = time.time()
  print('Starting image matching...')
  new = np.zeros((images[0].shape[0]*2, int(images[0].shape[1]*len(images)), 3))
  new.fill(-1)
  new[:images[0].shape[0], :images[0].shape[1], :] = np.array(images[0])
  fix_axis = [0,0]
  for i in range(new.shape[0]):
    if (-1 not in new[i, 5:images[0].shape[1]-5]):# and i >= fix_axis[0]:
      fix_axis[0] = i
      break
  for i in range(new.shape[0]-1, -1, -1):
    if (-1 not in new[i, 5:images[0].shape[1]-5]):# and i <= fix_axis[1]:
      fix_axis[1] = i
      break
  cv2.imwrite('stitch_template.jpg', new)
  new = cv2.imread('stitch_template.jpg')
  if args.plot:
    cv2.imshow('new', new)
    cv2.waitKey(0)
  os.system("mkdir processed_image")
  base = np.vstack([[0.0],[0.0],[1.0]])
  right_most = 0
  for i in tqdm(range(len(goods))):
    aff = Aff_trans(goods[i])
    xprime, yprime, y, x = imwarp(images[i+1], aff)
    original_coor = zip(x,y)
    new_coor = zip(xprime, yprime)
    new, boundy, boundx = blending(new, images[i+1], original_coor, new_coor, 30, base)
    for j in range(boundx[0], boundx[1]+1):
      if (-1 not in new[j, boundy[0]:boundy[1]]) and j >= fix_axis[0]:
        fix_axis[0] = j
        break
    for j in range(boundx[1], boundx[0]-1, -1):
      if (-1 not in new[j, boundy[0]:boundy[1]]) and j <= fix_axis[1]:
        fix_axis[1] = j
        break
    right_most = boundy[1]
    cv2.imwrite('processed_image/'+str(i)+'.jpg', new)
    new = cv2.imread('processed_image/'+str(i)+'.jpg')
    if args.plot:
      cv2.imshow('new', new)
      cv2.waitKey(0) 
    base = np.dot(aff, base)
    base[2][0] = 1.0
  print('Matching complete! time:', time.time()-t)
  total_time += time.time()-t

  print('Cropping...')
  t = time.time()
  new = new[fix_axis[0]+50:fix_axis[1], 0:right_most, :]
  print('Cropping done in ', time.time()-t)
  total_time += time.time()-t

  cv2.imwrite(args.out, new.astype(np.uint8))
  print('Total time:', total_time)


if __name__ == '__main__':
  main()