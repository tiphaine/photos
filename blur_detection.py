import click
import cv2
import imutils

from os import listdir, mkdir
from os.path import exists
from pprint import pprint
from shutil import copyfile


@click.command()
@click.argument('path')
@click.option('--threshold', '-t', default=100)
@click.option('--format', default='jpg')
@click.option('--destination', '-d', default='blur_detection')
@click.option('--sep', default='/')
def detect(path, threshold, format, destination, sep):
	clean_folder = '{}/clean'.format(destination)
	blurry_folder = '{}/blurry'.format(destination)
	folders2check = [destination, blurry_folder, clean_folder]
	for folder in folders2check:
		if not exists(folder):
			mkdir(folder)
	blurriness = {}
	photopaths = ['{}/{}'.format(path, file) for file in listdir(path) if file.lower().endswith(format)]
	with click.progressbar(photopaths) as bar:
		for photopath in bar:
			image = cv2.imread(photopath)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			blur_value = variance_of_laplacian(gray)
			blurriness[photopath] = blur_value

			values = photopath.split(sep)
			photo_name = values[-1]
			if blur_value < threshold:
				copyfile(photopath, '{}/{}'.format(blurry_folder, photo_name))
			else:
				copyfile(photopath, '{}/{}'.format(clean_folder, photo_name))
	pprint(blurriness)


def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()



if __name__ == '__main__':
	detect()
