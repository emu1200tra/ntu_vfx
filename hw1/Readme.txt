---run main python program for hdr creation---

Input image folder should contain a JPG folder to store JPEG images and a CR2 folder to store raw images.

python ./hw1.py --img_path [path/to/image/folder] --output_path [path/to/output/folder] --output_name [output name of .hdr image] --tone_output_name [output name of tone mapping image] --color_output_name [output name of post processing image] --resize [resize input image with the factor] --iteration [times to do color temperature transform] --iteration_sharp [times to do sharpening] --bright_factor [brightness enhancement factor] --key [key for tone mapping algorithm]

For example, to run our image set
python ./hw1.py --img_path ./input1 --output_path ./result1 --output_name result.hdr --tone_output_name result_tm.jpg --color_output_name result_color.jpg --resize 2 --iteration 3 --iteration_sharp 3 --bright_factor 3 --key 0.18

Requrement:

numpy
opencv-python
PIL
rawpy
skimage

---Run tome mapping software---

.\tm_photographic.exe -i [input hdr] -o [output jpg] -key [key]

For example, to transform our hdr reuslt
.\tm_photographic.exe -i ./result1/result.hdr -o ./result1/result_tm.jpg -key 0.4


