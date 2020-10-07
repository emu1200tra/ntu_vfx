import cv2
import numpy as np
import math
import sys
from scipy.interpolate import RectBivariateSpline
import scipy
from scipy.ndimage import filters
from scipy import signal,stats
import os

def blending(image1, image2, pold, pnew, constance, base):
  ratio = np.ones(image2.shape[1])
  ratio1 = np.linspace(0.0, 1.0, constance)
  ratio[:ratio1.shape[0]] = ratio1
  boundy = [2147483647, 0]
  boundx = [2147483647, 0]
  for i,j in zip(pold, pnew):
    shape1 = image1[int(j[0]+base[1][0]), int(j[1]+base[0][0]):int(j[1]+base[0][0])+2, :].shape
    shape2 = image2[int(i[0]), int(i[1]):int(i[1])+2, :].shape
    if shape1 == shape2:
      image1[int(base[1][0]+j[0]), int(base[0][0]+j[1]):int(base[0][0]+j[1])+2, :] = (1.0-ratio[int(i[1])])*image1[int(j[0]+base[1][0]), int(j[1]+base[0][0]):int(j[1]+base[0][0])+2, :] + ratio[int(i[1])]*image2[int(i[0]), int(i[1]):int(i[1])+2, :]
    if int(base[0][0]+j[1]) < boundy[0]:
      boundy[0] = int(base[0][0]+j[1])
    if int(base[0][0]+j[1]) > boundy[1]:
      boundy[1] = int(base[0][0]+j[1])
    if int(base[1][0]+j[0]) < boundx[0]:
      boundx[0] = int(base[1][0]+j[0])
    if int(base[1][0]+j[0]) > boundx[1]:
      boundx[1] = int(base[1][0]+j[0])
  return image1, boundy, boundx

def Aff_trans(good):
  n = len(good)
  p1 = [i[1] for i in good]
  p1 = np.array(p1).T
  p0 = [i[2] for i in good]
  p0 = np.array(p0).T
  est_outlier_rate = 0.5

  num_guesses = int(math.ceil(1.0/math.pow(1.0-est_outlier_rate, 3)))
  best_seed = np.zeros((3,1))
  best_median = np.inf
  best_aff = np.zeros((3,1))
  for k in range(num_guesses):
    c = [int(math.ceil(np.random.rand()*(n-1)))]
    for j in range(2):
      k = math.ceil(np.random.rand()*(n-1))
      while k in c:
        k = math.ceil(np.random.rand()*(n-1))
      c.append(int(k))
    A = []
    b = []
    for j in range(3):
      x = p0[0, c[j]]
      y = p0[1, c[j]]
      A.append([x, 0, y, 0, 1, 0])
      A.append([0, x, 0, y, 0, 1])
      b.append(p1[0,c[j]])
      b.append(p1[1,c[j]])
    aff = np.dot(np.linalg.pinv(np.vstack(A)), np.vstack(b))
    aff = np.asarray([[aff[0,0], aff[2,0], aff[4,0]], [aff[1,0], aff[3,0], aff[5,0]], [0, 0, 1]])
    tmp = np.empty((1,n))
    tmp.fill(1)
    pts = np.dot(aff, np.vstack((p0, tmp)))
    tmp = np.vstack(pts[0:2,:]-p1)
    resid = np.sqrt(np.sum(np.power(tmp, 2),0))

    if np.median(resid) < best_median:
      best_seed = c
      best_median = np.median(resid)
      best_aff = aff
  aff = best_aff

  for k in range(10):
    tmp = np.empty((1,n))
    tmp.fill(1)
    pts = np.dot(aff, np.vstack((p0, tmp)))

    tmp = np.vstack(pts[0:2,:]-p1)
    resid = np.sqrt(np.sum(np.power(tmp, 2),0)).T
    sigma = 1.4826*np.median(resid)
    dev = resid/sigma
    resid_wght = np.exp(-np.power(dev, 2)/2.0)/(math.sqrt(2.0*math.pi)*sigma)
    resid_wght[np.where(dev>5.0)] = 0.0
    A = []
    b = []
    for j in range(n):
      x = p0[0, j]
      y = p0[1, j]
      tmp = [resid_wght[j] * i for i in [x, 0, y, 0, 1, 0]]
      A.append(tmp)
      tmp = [resid_wght[j] * i for i in [0, x, 0, y, 0, 1]]
      A.append(tmp)
      b.append(resid_wght[j]*p1[0,j])
      b.append(resid_wght[j]*p1[1,j])
    aff = np.dot(np.linalg.pinv(np.vstack(A)), np.vstack(b))
    aff = np.asarray([[aff[0,0], aff[2,0], aff[4,0]], [aff[1,0], aff[3,0], aff[5,0]], [0, 0, 1]])

  return aff



def imwarp(im, A):
  eps = sys.float_info.epsilon
  wIm = np.zeros(im.shape)
  x, y = np.meshgrid(range(0,wIm.shape[1]),range(0,wIm.shape[0]))
  homogeneousCoords = np.vstack([x.T.flatten().tolist(), y.T.flatten().tolist(), [1 for i in range(wIm.shape[0]*wIm.shape[1])]])
  warpedCoords = np.dot(A, homogeneousCoords)
  
  yprime=warpedCoords[0,:].T

  xprime=warpedCoords[1,:].T
  return xprime, yprime, x.T.flatten(), y.T.flatten()


def read_img(file_path, txt_path):
  images = []
  focal = []
  with open(os.path.join(file_path, txt_path), 'r') as f:
    counter = 0
    for line in f:
      if counter % 2 == 0:
        print(line[:-1])
        img = cv2.imread(os.path.join(file_path, line[:-1]))
        images.append(img)
      else:
        focal.append(float(line[:-1]))
      counter+=1
  return images, focal

def prepare_image(images, focal):
  images_data = []
  for index in range(len(images)):
    center = (images[index].shape[0]//2, images[index].shape[1]//2)
    result = np.zeros(images[index].shape)
    for i in range(-images[index].shape[0]//2,images[index].shape[0]//2):
      for j in range(-images[index].shape[1]//2,images[index].shape[1]//2):
        y = int(round(focal[index]*math.atan(j/focal[index])))
        x = int(round(focal[index]*(i/math.sqrt(j*j+focal[index]*focal[index]))))
        result[center[0]+x,center[1]+y,:] = np.array(images[index][center[0]+i,center[1]+j,:])
    border = 10
    for i in range(result.shape[1]):      
      if int(np.sum(result[:,i,:])) != 0:
        border = i
        break
    result = np.array(result[100:-100, border:-border, :])
    images_data.append(result)
  return images_data

from tqdm import tqdm

def feature_cmp(in_para):

  desc_1, desc_2, pos_1, pos_2, ratio = in_para

  good = []
  for i in tqdm(range(desc_1.shape[0])):
    record = []
    for j in range(desc_2.shape[0]):
      dist = math.sqrt(np.sum(np.power(desc_1[i]-desc_2[j], 2)))
      record.append((dist, pos_1[i], pos_2[j]))
    record.sort(key=lambda x: x[0])
    if record[0][0] < record[1][0]*ratio:
      good.append(record[0])
      
  return good


def harris(im, sigma=3):
  size=im.shape
  X = np.zeros(size)
  Y = np.zeros(size)
  filters.gaussian_filter(im, sigma, (0,1), X)
  filters.gaussian_filter(im, sigma, (1,0), Y)
  xx = filters.gaussian_filter(X*X, sigma)
  xy = filters.gaussian_filter(X*Y, sigma)
  yy = filters.gaussian_filter(Y*Y, sigma)
  det = xx*yy-np.power(xy, 2)
  trc = xx+yy
  Res = det - 0.04*(np.power(trc, 2))
  Res=np.pad(Res[15:-15,15:-15], pad_width=15, mode='constant', constant_values=0)
  index = size[1]
  maxi = filters.maximum_filter(Res, 5)
  Res *= (Res==maxi)
  sorting = np.argsort(Res.ravel())[::-1][:512]
  pos=np.dstack(np.unravel_index(np.argsort(Res.ravel()), size))[:,:-513:-1,::-1].reshape([512,2])
  desc = []
  for j,i in pos:
    feature = im[i-15:i+16, j-15:j+16].flatten()
    desc.append(stats.zscore(feature))
  return (np.array(pos), np.array(desc))

  

  


