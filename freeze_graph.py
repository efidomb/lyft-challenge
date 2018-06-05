import tensorflow as tf
from tensorflow.python.framework import graph_util as tf_graph_util
import os
import shutil

def freeze_graph(element):
    # based on https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc

    ckpt_dir = 'saved_model/' + element
    frozen_model_dir = 'frozen_model/' + element

    checkpoint = tf.train.get_checkpoint_state(ckpt_dir) # get_latest_checkpoint_number
    input_checkpoint = checkpoint.model_checkpoint_path
    print("freezing from {}".format(input_checkpoint))
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    print("{} ops in the input graph".format(len(input_graph_def.node)))

    output_node_names = ["logits"]

    # freeze graph
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        # use a built-in TF helper to export variables to constants
        output_graph_def = tf_graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names
        )

    print("{} ops in the frozen graph".format(len(output_graph_def.node)))

    if os.path.exists(frozen_model_dir):
        shutil.rmtree(frozen_model_dir)

    # save model in same format as usual
    print('saving frozen model as saved_model to {}'.format(frozen_model_dir))
    # model = fcn8vgg16.FCN8_VGG16(define_graph=False)
    tf.reset_default_graph()
    tf.import_graph_def(output_graph_def, name='')
    with tf.Session() as sess:
        builder = tf.saved_model.builder.SavedModelBuilder(frozen_model_dir)
        builder.add_meta_graph_and_variables(sess, ['FCN'])
        builder.save()
        # model.save_model(sess, args.frozen_model_dir)

    print('saving frozen model as graph.pb (for transforms) to {}'.format(frozen_model_dir))
    with tf.gfile.GFile(frozen_model_dir+'/graph.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())


def optimise_graph(element):
    frozen_model_dir = 'frozen_model/' + element
    optimised_model_dir = 'optimised_model/' + element
    """ optimize frozen graph for inference """

    print('calling c++ implementation of graph transform')
    os.system('sh optimise.sh {}'.format(frozen_model_dir))

    # reading optimised graph
    tf.reset_default_graph()
    gd = tf.GraphDef()
    output_graph_file = frozen_model_dir+"/graph.pb" #optimised_graph
    with tf.gfile.Open(output_graph_file, 'rb') as f:
        gd.ParseFromString(f.read())
    tf.import_graph_def(gd, name='')
    print("{} ops in the optimised graph".format(len(gd.node)))

    # save model in same format as usual
    shutil.rmtree(optimised_model_dir, ignore_errors=True)
    #if not os.path.exists(args.optimised_model_dir):
    #    os.makedirs(args.optimised_model_dir)

    print('saving optimised model as saved_model to {}'.format(optimised_model_dir))
    tf.reset_default_graph()
    tf.import_graph_def(gd, name='')
    with tf.Session() as sess:
        builder = tf.saved_model.builder.SavedModelBuilder(optimised_model_dir)
        builder.add_meta_graph_and_variables(sess, ['FCN'])
        builder.save()
    shutil.move(frozen_model_dir+'/graph.pb', optimised_model_dir)#optimised_graph

element = 'roads'
freeze_graph(element)
optimise_graph(element)
element = 'vehicles'
freeze_graph(element)
optimise_graph(element)