import os
from glob import glob
import sys

camera_list = ['CameraRGB', 'CameraSeg']
# camera_list = ['rgb', 'seg']
folder = sys.argv[1]
for episode in os.listdir(folder):
	for camera in camera_list:
		path = os.path.join(folder, episode, camera)
		for image_file in glob(os.path.join(path, '*.png')):
			# print ('folder, episode, camera, path, image_file', folder, episode, camera, path, image_file)
			to = os.path.join(path, episode + '-' + os.path.basename(image_file))
			# print ('from, to', image_file, to)
			os.rename(image_file, to)