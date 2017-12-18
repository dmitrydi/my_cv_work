from .deep_sort.generate_detections import *
import numpy as np
import tensorflow as tf
import os
import sys
import tensorflow.python.platform
from tensorflow.python.platform import gfile

class DeepSortExtractor(object):
# класс для получения низкоразмерных свойств изображений в модели Deep Sort

  def __init__(self, model_path, batch_size=32, loss_mode='cosine'):
    self.model_path = model_path
    self.batch_size = batch_size
    self.loss_mode = loss_mode
    self.encoder = create_image_encoder(self.model_path, batch_size=self.batch_size, loss_mode=self.loss_mode)


  def get_bbox(sh):
    return (0,0,sh[1],sh[0])


  def encode(self, img_ary):
    def get_bbox(sh):
      return (0,0,sh[1],sh[0])
      
    cnn_shape = (128, 64)
    sh = img_ary.shape

    if len(sh) == 3:
      if sh[0] != cnn_shape[0] or sh[1] != cnn_shape[1]:
        img_ary_ = extract_image_patch(img_ary, get_bbox(sh), cnn_shape)[np.newaxis]
      else:
        img_ary_ = img_ary[np.newaxis]
    elif len(sh) == 4:
      img_ary_=None
      for img in img_ary:
        sh_=img.shape
        if sh_[0] != cnn_shape[0] or sh[1] != cnn_shape[1]:
          bbox = get_bbox(sh_)
          resized_img = extract_image_patch(img, bbox, cnn_shape)
        else:
          resized_img = img
        if img_ary_ is None:
          img_ary_ = resized_img[np.newaxis]
        else:
          img_ary_ = np.concatenate((img_ary_, resized_img[np.newaxis]))

    return self.encoder(img_ary_)


  def encode_from_dirs(self, root_path):
    def get_bbox(sh):
      return (0,0,sh[1],sh[0])

    labels = np.array([], dtype='int')
    imgs = None
    cnn_shape = (128,64)
    for dirname in os.listdir(root_path):
      for img_ in os.listdir(os.path.join(root_path, dirname)):
        img = cv2.imread(os.path.join(root_path, dirname, img_))
        sh = img.shape
        img = extract_image_patch(img, get_bbox(sh), cnn_shape)
        if imgs is None:
          imgs = img[np.newaxis]
        else:
          imgs = np.concatenate((imgs, img[np.newaxis]))
        labels = np.append(labels, int(dirname))
    features = self.encoder(imgs)
    del imgs

    return labels, features

class InceptionExtractor(object):

  def __init__(self, model_path):
    self.model_path = model_path

  def create_graph(self):
    """
    create_graph loads the inception model to memory, should be called before
    calling extract_features.
    model_path: path to inception model in protobuf form.
    """
    with gfile.FastGFile(self.model_path, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      _ = tf.import_graph_def(graph_def, name='')


  def extract_features(self, image_paths, verbose=False):
    feature_dimension = 2048
    features = np.empty((len(image_paths), feature_dimension))

    with tf.Session() as sess:
      flattened_tensor = sess.graph.get_tensor_by_name('pool_3:0')

      for i, image_path in enumerate(image_paths):
        if verbose:
          print('Processing %s...' % (image_path))

        if not gfile.Exists(image_path):
          tf.logging.fatal('File does not exist %s', image)

        image_data = gfile.FastGFile(image_path, 'rb').read()
        feature = sess.run(flattened_tensor, {'DecodeJpeg/contents:0': image_data})
        features[i, :] = np.squeeze(feature)

    return features


  def encode(self, image, verbose=False):
    tf.reset_default_graph()
    self.create_graph()
    image_path = [image]
    return self.extract_features(image_path,verbose=verbose)


  def encode_from_dirs(self, root_path, verbose=False):
    tf.reset_default_graph()
    self.create_graph()

    labels = np.array([], dtype='int')
    all_features = None

    for img_folder in os.listdir(root_path):
      if verbose:
        print('processing dir: {}'.format(img_folder))
      dummy = np.array([int(img_folder)],dtype='int')
      labels = np.append(labels, np.repeat(dummy,len(os.listdir(os.path.join(root_path, img_folder)))))
      img_paths = [os.path.join(root_path, img_folder, img) for img in os.listdir(os.path.join(root_path, img_folder))]
      features = self.extract_features(img_paths)
      if all_features is None:
        all_features = features
      else:
        all_features = np.concatenate((all_features, features))

    return labels, all_features

