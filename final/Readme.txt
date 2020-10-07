Usage:

##Reconstruction##
python ./patchMatch.py  --input [target image path] --ref [reference image path] --output [output name] --workers [num workers] --psize [patch size]

--example--
python ./patchMatch.py --input ./Sea_small.jpg --ref ./temple_small.jpg --output ./result.jpg --workers 4 --psize 7

##hole filling w/o constraint##
python ./update.py  --input [target image path] --ref [reference image path] --output [output name]  --hole [path to hole mask] --constraint None --workers [num workers] --stride [stride length] --psize [patch size]

--example--
python ./update.py  --input ./temple_small.jpg --ref ./temple_small.jpg --output ./test.jpg  --hole ./temple_mask1.png --constraint None --workers 8 --stride 5 --psize 21

##hole filling w/ constraint##
python ./update.py  --input [target image path] --ref [reference image path] --output [output name]  --hole [path to hole mask] --constraint [path to constraint] --workers [num workers] --stride [stride length] --psize [patch size]

--example--
python ./update.py  --input ./temple_small.jpg --ref ./temple_small.jpg --output ./test.jpg  --hole ./temple_mask2.png --constraint ./temple_const2.png --workers 8 --stride 5 --psize 21

##requirement##
opencv
numpy
multiprocessing
tqdm
PIL
