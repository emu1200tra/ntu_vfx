--how to run--
	python main.py --dir [path/to/image/dir] --txt [name/of/txt/file] --out [name/of/output/image] --plot [plot/image/or/not] --showP [show/special/debug/image]

for example: python .\main.py --dir ./modified/ --txt pano.txt --out ./panorama.jpg

--requirement--
opencv
numpy
multiprocessing
tqdm

--note--
text file should contain all the images' name and corresponding focal length. See txt file in imgs4 and modified for example.