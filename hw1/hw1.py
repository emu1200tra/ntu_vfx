import numpy as np
import cv2
import argparse
from PIL import Image 
from PIL.ExifTags import TAGS 
import os
import math
import random
import rawpy
import os
from colorTemp import *
from PIL import Image, ImageEnhance, ImageFilter
from skimage.restoration import denoise_tv_chambolle

def denoise(img):
    out = denoise_tv_chambolle(img, weight=0.6, multichannel=True)
    return out

def read_images(inputPathRaw, inputPathJpg, files, factor=1):
    images = []
    for i in files:
        path = i + ".JPG"
        img = Image.open(os.path.join(inputPathJpg,path))
        exif = {
          TAGS[k]: v
          for k, v in img._getexif().items()
          if k in TAGS
        }
        (a,b) = exif['ExposureTime']
        path = i + ".CR2"
        img = rawpy.imread(os.path.join(inputPathRaw, path))
        img = img.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)
        img = cv2.resize(img, (img.shape[1]//factor, img.shape[0]//factor))
        img = denoise(img.astype(np.float)).astype(np.int)
        print(img.shape)
        images.append((float(a)/float(b), img))
    images.sort(key=lambda x: x[0])
    imgs = [i[1] for i in images]
    info = [i[0] for i in images]
    return info, imgs

def get_file_list(inputPath):
    files = []
    imgs = os.listdir(inputPath)
    for i in imgs:
        name = i.split('.')[0]
        files.append(name)
    return files


def compute_irradiance(info, imgs):
    energy = np.zeros(imgs[0].shape)
    time = 0
    for i in range(len(info)):
        energy += (imgs[i]*info[i])
        time += info[i]**2
    return energy / time

def hdr(output_name, image):
    f = open(output_name, "wb")
    header = "#?RADIANCE\n# rerorerorerowryyyyyyyyyyy\nFORMAT=32-bit_rle_rgbe\n\n"
    header = header.encode()
    f.write(header)
    header = "-Y {0} +X {1}\n".format(image.shape[0], image.shape[1])
    header = header.encode()
    f.write(header)
    brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
    mantissa = np.zeros_like(brightest)
    exponent = np.zeros_like(brightest)
    np.frexp(brightest, mantissa, exponent)

    scaled_mantissa = mantissa * 256 / brightest
    scaled_mantissa[np.where(np.isnan(scaled_mantissa))] = 255.0
    scaled_mantissa[np.where(np.isinf(scaled_mantissa))] = 255.0
    rgbe = np.zeros((image.shape[0], image.shape[1], 4))
    rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
    rgbe[...,3] = np.around((exponent + 128))
    rgbe.astype(np.uint8).flatten().tofile(f)
    f.close()        

def parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--img_path", type=str, default='./image/', help="path of images")
  parser.add_argument("--output_path", type=str, default="./results", help="output path")
  parser.add_argument("--output_name", type=str, default="result.hdr", help="output file name")
  parser.add_argument("--tone_output_name", type=str, default="result_tm.jpg", help="output tone mapping file name")
  parser.add_argument("--color_output_name", type=str, default="result_color.jpg", help="output color temperature file name")
  parser.add_argument("--resize", type=int, default=2, help="resize factor")
  parser.add_argument("--iteration", type=int, default=3, help="iteration num for color temperature tunning")
  parser.add_argument("--iteration_sharp", type=int, default=3, help="iteration num for sharpening")
  parser.add_argument("--bright_factor", type=float, default=3, help="Brightness factor")
  parser.add_argument("--key", type=float, default=0.18, help="key for tone mapping")
  args = parser.parse_args()
  return args

def luminance(hdrImg):
    r = hdrImg[...,0]
    g = hdrImg[...,1]
    b = hdrImg[...,2]
    luminance_map = 0.27 * r + 0.67 * g + 0.06 * b
    return luminance_map

def tone_mapper(hdrImg, a, path):
    h,w,c = hdrImg.shape
    output = cv2.normalize(hdrImg, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    Image.fromarray(output.astype(np.uint8)).save(path)
    for i in range(c):
        hdrImg[...,i] = (hdrImg[...,i]-np.amin(hdrImg[...,i]))/(np.amax(hdrImg[...,i]-np.amin(hdrImg[...,i])))
    pixels = h*w
    delta = 0.0001
    luminance_map = luminance(hdrImg)
    key = np.exp((1 / pixels)*np.sum(np.log(delta + luminance_map)))
    scaled_luminance = (a / key) * luminance_map
    final_luminance = scaled_luminance*(1+(scaled_luminance/(np.amax(scaled_luminance))**2)) / (1 + scaled_luminance)
    output = (hdrImg / luminance_map[...,None]) * final_luminance[...,None]
    output = cv2.normalize(output, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return output

def post_process(image, output_path, color_output_path, iteration=2, bright_factor=1.2, iteration_sharp=3):
    img = Image.fromarray(image.astype(np.uint8))
    img.save(output_path)
    for i in range(iteration):
        img = convert_temp(img, 8000)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(bright_factor)
    for i in range(iteration_sharp):
        img = img.filter(ImageFilter.SHARPEN);
    img.save(color_output_path)
    return img

'''
def mapper(input_path, output_path, color_output_path, iteration = 2, bright_factor = 1.2, key = 0.4):
    os.system('tm_photographic -i '+input_path+' -o '+output_path+' -key '+str(key))
    img = Image.open(output_path)
    for i in range(iteration):
        img = convert_temp(img, 8000)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(bright_factor)
    for i in range(3):
        img = img.filter(ImageFilter.SHARPEN);
    img.save(color_output_path)
    return img
'''

def main():
    opt = parser()
    files = get_file_list(os.path.join(opt.img_path, 'JPG'))
    info, imgs = read_images(os.path.join(opt.img_path, 'CR2'), os.path.join(opt.img_path, 'JPG'), files, opt.resize)
    irradiance_img = compute_irradiance(info, imgs)
    if not os.path.isdir(opt.output_path):
        os.mkdir(opt.output_path)
    hdr(os.path.join(opt.output_path, opt.output_name), irradiance_img)
    print('HDR done...processing tone mapping')
    output = tone_mapper(irradiance_img, opt.key, os.path.join(opt.output_path, opt.output_name.split('.')[0]+'.jpg'))
    post_process(output, os.path.join(opt.output_path, opt.tone_output_name), os.path.join(opt.output_path, opt.color_output_name), opt.iteration, opt.bright_factor, opt.iteration_sharp)
    #img = mapper(os.path.join(opt.output_path, opt.output_name), os.path.join(opt.output_path, opt.tone_output_name), os.path.join(opt.output_path, opt.color_output_name), opt.iteration, opt.bright_factor, opt.key)

if __name__ == "__main__":
    main()