import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time
import numpy as np
import warnings
from glob import glob
import re
import scipy.misc
from upload import TransferData
import threading
import shutil
from shutil import copyfile
warnings.filterwarnings("ignore")

transferData = TransferData()

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    pass
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    w3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    w4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    w7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    
    return w1, keep, w3, w4, w7
# tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    conv_7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
        kernel_initializer = tf.truncated_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    first_upsampling = tf.layers.conv2d_transpose(conv_7, num_classes, 4, 2, padding='same',
        kernel_initializer = tf.truncated_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    conv_4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
        kernel_initializer = tf.truncated_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    first_skip = tf.add(first_upsampling, conv_4)
    second_upsampling = tf.layers.conv2d_transpose(first_skip, num_classes, 4, 2, padding='same',
        kernel_initializer = tf.truncated_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    conv_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
        kernel_initializer = tf.truncated_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    second_skip = tf.add(second_upsampling, conv_3)
    output = tf.layers.conv2d_transpose(second_skip, num_classes, 16, 8, padding='same',
        kernel_initializer = tf.truncated_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return output
# tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.d
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
    correct_label = tf.identity(correct_label, name="correct_label")
    # correct_label = tf.reshape(correct_label, (-1, num_classes), name='correct_label') # NOT SURE ABOUT THIS LINE
#     weights = [3, 7]
#     weights = np.array(weights, dtype=np.int64)
#     tf.sparse_placeholder(tf.int32, shape=weights)
#     pos_weight = tf.constant(3, tf.float32)
    # weights = tf.gather(logits, class_weights)
    # cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
#     weighted_logits = tf.nn.weighted_cross_entropy_with_logits(logits=nn_last_layer, targets=correct_label, pos_weight=pos_weight)
#     cross_entropy_loss = tf.reduce_mean(logits)
    # cross_entropy_loss = tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label), class_weights))
    softmax = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)
#     logits = nn_last_layer
#     softmax = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=correct_label)
    
    cross_entropy_loss = tf.reduce_mean(softmax, name='cross_entropy_loss')
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss, name='train_op')
    return logits, train_op, cross_entropy_loss, nn_last_layer, softmax # nn_last_layer and forword should be remove
# tests.test_optimize(optimize)

def gen_output(sess, correct_label, cross_entropy_loss, keep_prob, image_pl, image_shape, crops):
    test_data_dir = os.path.join('Train', element, 'testing')
    # image_paths = glob(os.path.join(test_data_dir, 'images', '*.png'))
    image_paths = glob('Train/roads/testing/images/*.png')
    label_paths = {
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob(os.path.join(test_data_dir, 'gt_images', '*.png'))}
    background_color = np.array([0, 0, 255])
    counter = 0
    for image_file in image_paths:
        if counter >= restrict_testing:
            break
        counter += 1
        image = scipy.misc.imread(image_file)
        image = image[crops[0]:crops[1]]
        image = scipy.misc.imresize(image, image_shape)

        gt_image_file = label_paths[os.path.basename(image_file)]
        gt_image = scipy.misc.imread(gt_image_file)
        gt_image = gt_image[crops[0]:crops[1]]
        gt_image = scipy.misc.imresize(gt_image, image_shape)
        gt_bg = np.all(gt_image == background_color, axis=2)
        gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
        gt_image = np.concatenate((gt_bg, np.invert(gt_bg)*1.5), axis=2)
        feed_dict = {correct_label: [gt_image], keep_prob: 1.0, image_pl: [image]}
        loss = sess.run([cross_entropy_loss], feed_dict)
        yield loss

def test_loss(sess, correct_label, cross_entropy_loss, keep_prob, input_image, image_shape, crops):
    print ('calculating losses for testing set...')
    losses = gen_output(sess, correct_label, cross_entropy_loss, keep_prob, input_image, image_shape, crops)
    losses_list = []
    for loss in losses:
        losses_list.append(loss[0])
    return sum(losses_list) / len(losses_list)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, logits, nn_last_layer, softmax, last_epoch_num):
    saver = tf.train.Saver(allow_empty=True)
    rounds = epochs * (restrict_training / float(batch_size))
    losses = []
    total_epochs = epochs + last_epoch_num
    t0 = time.time()
    t2 = time.time()
    print ('start training...')
    for epoch in range(last_epoch_num, total_epochs):
        position = 0
        for image, label in get_batches_fn(batch_size):
            feed_dict={input_image: image, correct_label: label, keep_prob: 0.6, learning_rate: 0.001}
            _, loss, log, last_layer, soft = sess.run([train_op, cross_entropy_loss, logits, nn_last_layer, softmax], feed_dict=feed_dict)
            position += batch_size
            losses.append(loss)
            average_loss = None
            if len(losses) > 20: average_loss = sum(losses[-20:]) / 20
            t1 = time.time()
            rounds -= 1
            estimated_time = (t1 - t0) * rounds
            print ('loss: ', loss, ' | epoch:', epoch, ' | position:', position, ' | ', estimated_time / 60, 'minutes left | average loss(20):', average_loss)
            t0 = time.time()
        t3 = time.time()
        if (t3-t2) > 1800:
        # if epoch % 1 == 0: # change it later to something bigger
            print('Saving model for epoch:', epoch)
            saver.save(sess, model_path, global_step=epoch, write_meta_graph=True)
            for file in os.listdir(save_dir):
                if file != ('model-' + str(epoch) + '.data-00000-of-00001'):
                    if file != ('model-' + str(epoch) + '.index'):
                        if file != ('model-' + str(epoch) + '.meta'):
                            if file != 'checkpoint':
                                os.remove(os.path.join(save_dir, file))

            print ('making zip file...')
            # shutil.make_archive('copy/' + element + '-' + str(epoch), 'zip', save_dir)
            # t = threading.Thread(target=upload_to_dropbox, args=(epoch,))
            # t.start()
            t2 = time.time()
        average_test_loss = test_loss(sess, correct_label, cross_entropy_loss, keep_prob, input_image, image_shape, crops)
        print ('average loss for testing set:', average_test_loss)
        write_report(epoch, average_loss, average_test_loss)
    print('Saving model for epoch:', epoch)
    saver.save(sess, model_path, global_step=epoch, write_meta_graph=True)
    print ('making zip file...')
    # shutil.make_archive('copy/' + element + '-' + str(epoch), 'zip', save_dir)
    # t = threading.Thread(target=upload_to_dropbox, args=(epoch,))
    # t.start()
    print('--- Done ---\n')
#             print ('logits:', log) # or shape
#             print ('softmax', soft) # or shape
# tests.test_train_nn(train_nn)

def write_report(epoch, average_loss, average_test_loss):
    file = open('report.txt', 'a')
    text = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ' | '
    text += element + ' | epoch #' + str(epoch) + ': average_loss ' + str(average_loss) + ' | average_test_loss ' + str(average_test_loss) + '\n'
    file.write(text)
    file.close()
    transferData.upload_file('report.txt')

def upload_to_dropbox(epoch):
    print ('uploading saved model for epoch', epoch, 'to dropbox')
    t0 = time.time()
    transferData.upload_file('copy/' + element + '-' + str(epoch) + '.zip', delete_original=True)
    t1 = time.time()
    print ('uploading for epoch', epoch, 'is now done. time:', (t1-t0)/60, 'minutes')

def run(retrain=False):
    epochs = 10
    batch_size = 8
    tf.reset_default_graph()
    data_dir = os.path.join('./Train', element)
    with tf.Session() as sess:
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'training'), image_shape, crops, restrict_training)
        if retrain:
            latest_ckpt_name = tf.train.latest_checkpoint(save_dir)
            # for specific checkpoint or in case something went wrong with the latest_ckpt_name - do it manually
            # latest_ckpt_name = './saved_model/roads/model-' + str()

            print('latest ckpt is:', latest_ckpt_name)
            if latest_ckpt_name is not None:
                meta_graph_name = latest_ckpt_name + '.meta'
                print('looking for meta graph:', meta_graph_name)
                new_saver = tf.train.import_meta_graph(meta_graph_name)  

                print('restoring model from latest ckpt:', latest_ckpt_name)
                new_saver.restore(sess, latest_ckpt_name)
                last_epoch_num = helper.get_latest_checkpoint_number(save_dir)

                print('=== Continuing Training === (for epochs: ', epochs, ')')

                keep_prob = tf.get_collection('keep_prob')[0]
                input_image = tf.get_collection('input_image')[0]
                logits = tf.get_collection('logits')[0]
                correct_label = tf.get_collection('correct_label')[0]
                learning_rate = tf.get_collection('learning_rate')[0]
                cross_entropy_loss = tf.get_collection('cross_entropy_loss')[0]
                train_op = tf.get_collection('train_op')[0]
                nn_last_layer = tf.get_collection('nn_last_layer')[0]
                softmax = tf.get_collection('softmax')[0]
            else:
                print ('latest ckpt is None. check your directory.')
        else:
            correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])
            learning_rate = tf.placeholder(tf.float32)
            input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
            output = layers(layer3_out, layer4_out, layer7_out, num_classes)
            logits, train_op, cross_entropy_loss, nn_last_layer, softmax = optimize(output, correct_label, learning_rate, num_classes)
            last_epoch_num = 1

            # Add ops/tensors to collection - to save in the checkpoint for later use
            tf.add_to_collection('input_image', input_image)
            tf.add_to_collection('keep_prob', keep_prob)
            tf.add_to_collection('learning_rate', learning_rate)
            tf.add_to_collection('correct_label', correct_label)
            tf.add_to_collection('logits', logits)
            tf.add_to_collection('cross_entropy_loss', cross_entropy_loss)
            tf.add_to_collection('train_op', train_op)
            tf.add_to_collection('nn_last_layer', nn_last_layer)
            tf.add_to_collection('softmax', softmax)

            sess.run(tf.global_variables_initializer())
        
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, logits, nn_last_layer, softmax, last_epoch_num)
        
        test_dir = os.path.join(data_dir, 'testing', 'images')
        # helper.pred_samples(runs_dir, test_dir, sess, image_shape, logits, keep_prob, input_image, crops, restrict_prediction)

def predict_images(test_data_path=None, video=None):
    save_dir = os.path.join('saved_model', element)
    tf.reset_default_graph()
    with tf.Session() as sess:
        # Predict the logits
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits = tf.reshape(nn_last_layer, (-1, num_classes))

        saver = tf.train.Saver()
        last_checkpoint = helper.get_latest_checkpoint_number(save_dir) - 1
        if video is None:
            saver.restore(sess, os.path.join('saved_model', element, 'model-' + str(last_checkpoint)))
            print("Restored the saved Model in save_model")
            helper.pred_samples(runs_dir, test_data_path, sess, image_shape, logits, keep_prob, input_image, crops, restrict_prediction)
        else:
            save_dir = os.path.join('saved_model', test_data_path)
            last_checkpoint = helper.get_latest_checkpoint_number(save_dir) - 1
            saver.restore(sess, os.path.join('saved_model', test_data_path, 'model-' + str(last_checkpoint)))
            return helper.get_binary_seg(video, sess, image_shape, logits, keep_prob, input_image, crops)

def opt_predict_images(test_data_path=None, video=None):
    frozen_graph = os.path.join('optimised_model', element, 'graph.pb')
    # frozen_graph = os.path.join('frozen_model', element, 'saved_model.pb')
    tf.reset_default_graph()

    with tf.gfile.GFile(frozen_graph, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,
            input_map=None,
            return_elements=None,
            name=""
            )

    input_image = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    logits = graph.get_tensor_by_name('logits:0')
    sess = tf.Session(graph=graph)
    # with tf.Session(graph=graph) as sess:
    if video is None:
        helper.pred_samples(runs_dir, test_data_path, sess, image_shape, logits, keep_prob, input_image, crops, restrict_prediction)
    else:
        return helper.get_binary_seg(video, sess, image_shape, logits, keep_prob, input_image, crops)

element = 'roads'
# element = 'vehicles'
save_dir = os.path.join('saved_model', element) # where checkpoint will be saved # './saved_model'
model_name  = 'model'
model_path = os.path.join(save_dir, model_name)
image_shape = (160, 576) # other possibilities is (480, 576)
crops = (235, 460)
if element == 'roads':
    crops = (0, 600)
num_classes = 2
runs_dir = './runs'
vgg_path = os.path.join('./Train', 'vgg')
helper.maybe_download_pretrained_vgg('./Train')
restrict_training = 4640 # no more than 4640
restrict_testing = 300
restrict_prediction = 10

if __name__ == '__main__':
    
    run(retrain=True)
    
    # use the pre-trained model to predict more images
    test_data_path = 'Train/roads/testing/images'
#     predict_images('trial')
    # predict_images(test_data_path)
    # opt_predict_images(test_data_path)