import tensorflow as tf
import sys
import glob 
import os 
import cv2
import argparse

FLAGS = None

def main(_):
    image_root_path = FLAGS.test_image_folder # all testing images in this folder, e.g. 'test_image_folder/'
    model_root_path = FLAGS.models_folder # copy trained model in this folder. (retrained_graph.pb, retrained_labels.txt)
    if_display_image = FLAGS.display_image # display the current image if if_display_image=True

    path_list = glob.glob(os.path.join(image_root_path, '*.*'))
    temp_image_name = 'temp.jpg'
    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line in tf.gfile.GFile(os.path.join(model_root_path, 'retrained_labels.txt'))]

    # Unpersists graph from file
    with tf.gfile.FastGFile(os.path.join(model_root_path, 'retrained_graph.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        for image_path in path_list:
            print image_path
            if '.jpg' not in image_path[-4:]:
                n_image = cv2.imread(image_path)
                image_path = temp_image_name
                cv2.imwrite(image_path, n_image)

            # Read in the image_data
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()

            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            
            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                print('%s (score = %.5f)' % (human_string, score))
            print ''

            if if_display_image=='True':
                image = cv2.imread(image_path)
                cv2.imshow('image', image)
                cv2.waitKey(0)

        if os.path.exists(temp_image_name):
            os.remove(temp_image_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--models_folder',
      type=str,
      help='Path to folders of trained models.'
    )
    parser.add_argument(
      '--test_image_folder',
      type=str,
      help='Path to folders of testing images.'
    )
    parser.add_argument(
      '--display_image',
      type=str,
      default='False',
      help='Display testing images if its value is True'
    )
    FLAGS, unparsed = parser.parse_known_args()
    try:
        tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    except Exception as e:
        print '''\
        Usage: 

        python predict.py \\
            --models_folder='./models' \\
            --test_image_folder='./test_images' \\
            --display_image=False

        (Type "python predict.py -h" for more details)
        '''