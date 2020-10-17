import cv2
import numpy as np 
import os


def get_image_paths(directory):
    return [x.path for x in os.scandir(directory) if x.name.endswith(".jpg") or x.name.endswith(".png")
            or x.name.endswith(".JPG")]


def load_images(img_path,convert=None):
	all_imgs=(cv2.resize(cv2.imread(i),(256,256)) for i in img_path)
	if convert:
		all_imgs=(convert(img) for img in all_imgs)

	for i,img in enumerate(all_imgs):
		if i==0:
			all_imgs_final=numpy.empty((len(img_path),) + img.shape,dtype=img.dtype)
		all_imgs_final[i]=img
	print("shape of all_imgs_final",np.array(all_imgs_final).shape)
	return all_imgs_final




def get_transpose_axes(n):
    if n % 2 == 0:
        y_axes = list(range(1, n - 1, 2))
        x_axes = list(range(0, n - 1, 2))
    else:
        y_axes = list(range(0, n - 1, 2))
        x_axes = list(range(1, n - 1, 2))
    return y_axes, x_axes, [n - 1]


def stack_images(images):
    images_shape = numpy.array(images.shape)
    new_axes = get_transpose_axes(len(images_shape))
    new_shape = [numpy.prod(images_shape[x]) for x in new_axes]
    return numpy.transpose(
        images,
        axes=numpy.concatenate(new_axes)
    ).reshape(new_shape)