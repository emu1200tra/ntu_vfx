import os

cr2_path = "./cr2"
jpg_path = "./jpg"

files = os.listdir(cr2_path)
files.sort()

counter = 0
for file in files:
    name = file.split('.')[0]
    os.system('cp ./cr2/'+name+'.CR2 '+'./image/CR2')
    os.system('cp ./jpg/'+name+'.JPG '+'./image/JPG')
    counter += 1
    if counter == 3:
        counter = 0
        os.system('python hw1.py --output_name '+name+'.hdr')
        #os.system('./tm_photographic -i ./results/'+name+'.hdr -o ./results/'+name+'.jpg -key 0.5')
        os.system('rm ./image/CR2/*')
        os.system('rm ./image/JPG/*')
    print(file)
