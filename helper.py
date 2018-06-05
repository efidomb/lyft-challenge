import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
from PIL import Image
from io import BytesIO, StringIO
import sys, skvideo.io, json, base64
import cv2
from skimage.transform import resize

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(data_folder, image_shape, crops, restrict):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'images', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_images', '*.png'))}
        background_color = np.array([0, 0, 255]) # originaly: [255, 0, 0]
        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            counter = 0
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                if counter >= restrict:
                    break
                counter += 1
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file)[crops[0]:crops[1]], image_shape)
#                 image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file)[crops[0]:crops[1]], image_shape)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)*1.5), axis=2)

                images.append(image)
                gt_images.append(gt_image)
                
                augment = True
                if augment:
                    # shade
                    shade_image = image * 0.4 + np.random.rand() * 0.6
                    images.append(shade_image)
                    gt_images.append(gt_image)

#                     for i in range(3):
#                         colored_image = image[:, :, i] * 0.1 + np.random.rand()
#                         images.append(colored_image)
#                         gt_images.append(gt_image)

                    noise_image = image * np.ones(image.shape)*0.95 + np.random.rand(image.shape[0],image.shape[1],image.shape[2])*0.1
                    images.append(noise_image)
                    gt_images.append(gt_image)

                    flip_image = np.flip(image, 1)
                    images.append(flip_image)
                    flip_gt_image = np.flip(gt_image, 1)
                    gt_images.append(flip_gt_image)
            yield np.array(images), np.array(gt_images)
    return get_batches_fn

def segment(image, sess, image_shape, logits, keep_prob, image_pl, crops, transperent=True):
    hight = crops[1] - crops[0]
    if transperent:
        top = np.zeros((crops[0], 800, 4))
        bottom = np.zeros((600 - crops[1], 800, 4))
    else:
        top = np.zeros((crops[0], 800, 3))
        bottom = np.zeros((600 - crops[1], 800, 3))
    image = image[crops[0]:crops[1]]
    image = scipy.misc.imresize(image, image_shape)
    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, image_pl: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    segmentation = resize(segmentation, (hight, 800, 1))
    if transperent:
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    else:
        mask = np.dot(segmentation, np.array([[0, 255, 0]]))
    mask = np.vstack((top, mask, bottom))
    return mask

def gen_output(sess, logits, keep_prob, image_pl, data_folder, image_shape, crops, restrict):
    counter = 0
    image_paths = os.listdir(data_folder)
    for image_file in image_paths:
        image_file = os.path.join(data_folder, image_file)
        if counter >= restrict:
            break
        counter += 1
        image = scipy.misc.imread(image_file)
        startTime = time.clock()
        mask = segment(image, sess, image_shape, logits, keep_prob, image_pl, crops)
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)
        endTime = time.clock()
        speed = endTime - startTime

        yield os.path.basename(image_file), np.array(street_im), speed
        
def pred_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, crops, restrict):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print('Predicting images...')

    image_outputs = gen_output(sess, logits, keep_prob, input_image, data_dir, image_shape, crops, restrict)
    speeds = []
    for name, image, speed in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
        speeds.append(speed)
    fps = 1 / (sum(speeds) / len(speeds))

    print('All test images are saved to: {}.'.format(output_dir))
    print ('frames per second:', fps)
            
def get_binary_seg(video, sess, image_shape, logits, keep_prob, image_pl, crops):
    masks = []
    hight = crops[1] - crops[0]
    top = np.zeros((crops[0], 800, 3))
    bottom = np.zeros((600 - crops[1], 800, 3))
    counter = 0
    for rgb_frame in video:
        mask = segment(rgb_frame, sess, image_shape, logits, keep_prob, image_pl, crops, transperent=False)
        green = mask[:, :, 1]
        mask = np.where(green>254,1,0).astype('uint8')
        # scipy.misc.imsave('masks/'+str(counter)+'.png', mask)
        # scipy.misc.imsave('frames/'+str(counter)+'.png', rgb_frame)
        masks.append(mask)
        counter += 1
    return masks

def get_latest_checkpoint_number(save_dir):
    ''' Helper func. Return the latest checkpoint number from save_dir
    '''
    epoch_num = 0
    kvpairs = []
    checkpoint_file = os.path.join(save_dir, 'checkpoint')
    """ Note: 'checkpoint' file format is:
        model_checkpoint_path: "my-model-2"
        all_model_checkpoint_paths: "my-model-0"
        all_model_checkpoint_paths: "my-model-1"
    """
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rt') as inf:
            for ln in inf:
                ln = ln.strip('\n')
                ln = ln.replace("\"", '')
                ln = ln.replace(' ', '')
                k, v = ln.split(':')
                if k == 'model_checkpoint_path':
                    chk_value = v
                    break
        epoch_num = int(chk_value.split('-')[-1])
    
    return epoch_num + 1