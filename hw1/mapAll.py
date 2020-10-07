import os
from colorTemp import *
from PIL import Image, ImageEnhance 

path = "./results"

files = os.listdir(path)

for file in files:
    name = file.split('.')[0]
    os.system('tm_photographic -i ./results/'+name+'.hdr -o ./results/recovered_'+name+'.jpg -key 0.4')
    img = Image.open('./results/recovered_'+name+'.jpg')
    img = convert_temp(img, 25000)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.2)
    img.save('./results/recovered_'+name+'_colorTemp.jpg')