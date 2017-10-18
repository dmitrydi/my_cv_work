from keras.layers import Dense, Lambda, Input, merge, Reshape, Flatten
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import SGD
import keras.backend as K
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tpe_metrics import get_scores, calc_metrics
from tpe_model_builder import build_tpe
from sklearn.decomposition import PCA
import os
from keras import backend as K
K.set_image_dim_ordering('th')

def make_data_from_images(root_path, num_images, img_h, img_w, rescale_factor=255., batch_size=100):
# преобразует изобажения в папке root_path в векторное представление (3, img_h, img_w)
# PARAMS
# root_path - имя корневой директории с изображениями. Каждый класс изображений хранится в подпапке с уникальным именем
# num_images - общее число изображений для всех классов
# batch_size - размер батча для генерации 
# img_h - выходной размер изображения по y
# img_w - выходной размер изображения по x
# rescale_factor - фактор масштабирования канала цвета
# OUTPUT
# x_data = (num_images, 3, img_h, img_w)
# y_data = (num_images, n_classes)

    datagen = ImageDataGenerator(1/rescale_factor)
    imax = num_images//batch_size + int(bool(num_images%batch_size))
    x_data=None
    y_data=None
    i = 1
    
    for x_batch, y_batch in datagen.flow_from_directory(
            root_path,
            target_size=(img_h, img_w),
            batch_size=batch_size,
            shuffle=False):
        if x_data is None:
            x_data=x_batch
            y_data=y_batch
        else:
            x_data=np.vstack([x_data,x_batch])
            y_data=np.vstack([y_data,y_batch])
        i+=1
        if i>imax:
            break
            
    labels = np.array([], dtype=int)
    
    for y in y_data:
        labels = np.append(labels, np.nonzero(y)[0])
            
    return x_data, y_data, labels.flatten()

def make_embeddings(x_data, neural_net, out_layer_number=~2, batch_size=100):
# выполняет преобразование входных изображений x_data в вектора низкой размерности (bottlenecking)
# x_data - массив входных изображений (?, 3, img_h, img_w)
# neural_net - обученная нейронная сеть (вместе с последним слоем)
# out_layer_number - слой модели, с которого берутся векторные представления низкой размерности (embeddings)

    class Bottleneck:
        def __init__(self, model, layer):
            self.fn = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output])

        def predict(self, data_x, batch_size=100):
            n_data = len(data_x)
            n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
            result = None

            for i in range(n_batches):
                batch_x = data_x[i * batch_size:(i + 1) * batch_size]
                batch_y = self.fn([batch_x, 0])[0]
                if result is None:
                    result = batch_y
                else:
                    result = np.vstack([result, batch_y])

            return result
    
    x_emb = Bottleneck(neural_net, out_layer_number).predict(x_data, batch_size=batch_size)
    
    return x_emb

def get_batch(x_embs, tpe_predictor, labels, n_classes, batch_size, random_sample_size):
# mining hard negative examples here
# returns |(a, p, n)| == batch_size
# x_embs - эмбеддинги изображений в исходной нейросети со срезанным последним слоем
    # tpe_predictor : a*W
    # для каждого класса ищем максимально далекий positive:
    a, p, n = None, None, None
    n_elements = x_embs.shape[0]
    for _ in range(batch_size):
        ind = np.random.randint(0, n_elements, 1)
        x_anchor = x_embs[ind]
        x_anchor_label = labels[ind]
        x_positive = get_hard_positive(x_anchor, x_embs, tpe_predictor, labels, x_anchor_label, random_sample_size=random_sample_size)
        x_negative = get_hard_negative(x_anchor, x_embs, tpe_predictor, labels, x_anchor_label, random_sample_size=random_sample_size)
        if a is None:
            a = x_anchor
            p = x_positive
            n = x_negative
        else:
            a = np.vstack([a, x_anchor])
            p = np.vstack([p, x_positive])
            n = np.vstack([n, x_negative])

    return a,p,n

def get_hard_positive(x_anchor, x_emb_W, tpe_predictor, labels, class_label, random_sample_size=100):
    x_class = x_emb_W[labels==class_label]
    x_class_tpe = tpe_predictor.predict(x_class)
    x_anchor_tpe = tpe_predictor.predict(x_anchor)
    scores = x_anchor_tpe@x_class_tpe.T
    self_ind = np.where(np.all(x_class==x_anchor,axis=1))[0][0]
    rand_indices = np.random.rand(scores.shape[1])
    rand_indices[self_ind] = 1 # avoid choosing self-element
    rand_scores = scores*rand_indices # убираем возможность переобучения на каком-то одном hard - экземпляре
    ind = np.argmin(rand_scores) # ищем самый далекий экземпляр того же класса
    
    return x_class[ind]

def get_hard_negative(x_anchor, x_emb_W, tpe_predictor, labels, class_label, random_sample_size=100):
    x_not_class = x_emb_W[labels!=class_label]
    indices = np.random.choice(range(x_not_class.shape[0]), size=random_sample_size, replace=False)
    x_not_class = x_not_class[indices]
    x_not_class_tpe = tpe_predictor.predict(x_not_class)
    x_anchor_tpe = tpe_predictor.predict(x_anchor)
    scores = x_anchor_tpe@x_not_class_tpe.T
    rand_scores = scores*np.random.rand(scores.shape[1])
    ind = np.argmax(rand_scores) # ищем самый близкий экземпляр не из своего класса
    
    return x_not_class[ind]

def make_dev_data_and_protocol(root_path, num_images, img_h, img_w, rescale_factor=255., batch_size=100):
# создает разметку для development-множества
# root_path - корневая директория для development-множества
# num_images - количество изображений
# img_h
# img_w
# rescale_factor
# batch_size
    x_data, y_data, labels = make_data_from_images(
        root_path, num_images, img_h, img_w, rescale_factor=rescale_factor, batch_size=batch_size)
    
    protocol = np.matrix(np.zeros((labels.shape[0], labels.shape[0])), dtype=int)
    for label in np.unique(labels):
        label_mask = (labels == label)
        label_vect = np.matrix(np.ones(label_mask.shape)*label_mask, dtype=int)
        label_matr = label_vect.T*label_vect
        protocol += label_matr
        
    protocol = np.array(protocol, dtype=int)
        
    return x_data, y_data, labels, protocol

def train_tpe(
    tpe_model, tpe_predictor, x_embs, x_embs_dev, dev_protocol, labels,
    n_classes, nepoch, batch_size, sampling_batch_size, model_saving_path, model_saving_name):

    z = np.zeros((batch_size,))
    mineer = 1.
    saving_name = os.path.join(model_saving_path, model_saving_name)
    for _i in range(nepoch):
        print('epoch: {}'.format(_i))
        a, p, n = get_batch(x_embs, tpe_predictor, labels, n_classes, batch_size, random_sample_size=sampling_batch_size)
        tpe_model.fit([a,p,n], z, batch_size=batch_size, epochs=1)
        x_emb_dev_tpe = tpe_predictor.predict(x_embs_dev)
        tsc, isc = get_scores(x_emb_dev_tpe, dev_protocol) # обновление scores из dev_emb2  и dev_protocol. Что такое dev_protocol?
        eer, _, _, _, deer = calc_metrics(tsc, isc)
        print('EER: {:.2f}'.format(eer * 100))
        if eer < mineer:
            mineer = eer
            mindeer = deer
            tpe_model.save_weights(saving_name)
            
    return mineer, mindeer

def train_tpe_from_dirs(
    cnn_model, nepoch=50, num_images_train=3000, num_images_dev=600, train_path='../data/train', dev_path='../data/test', n_classes=30,  bottle_layer=~2,
    in_shape=64, out_shape=64, img_h=120, img_w=60, rescale_factor=255., data_batch_size=100, train_batch_size = 100, sampling_batch_size=100,
    tpe_in_shape=64, tpe_out_shape=64, model_saving_path='../models', model_saving_name='tpe_model.h5', eer_d_log_file='params_tpe.txt'):

    x_data, y_data, labels = make_data_from_images(train_path, num_images_train, img_h, img_w, rescale_factor=rescale_factor, batch_size=data_batch_size)
    x_embs = make_embeddings(x_data, cnn_model, out_layer_number=bottle_layer, batch_size=data_batch_size)
    x_dev, y_dev, labels_dev, protocol = make_dev_data_and_protocol(dev_path, num_images_dev, img_h, img_w, rescale_factor=rescale_factor, batch_size=data_batch_size)
    x_embs_dev = make_embeddings(x_dev, cnn_model, out_layer_number=bottle_layer, batch_size=data_batch_size)
    # инициализируем веса tpe-модели с помощью матрицы PCA
    pca = PCA(tpe_out_shape)
    pca.fit(x_embs)
    W_pca = pca.components_
    tpe_model, tpe_predictor = build_tpe(tpe_in_shape, tpe_out_shape, W_pca)
    mineer, mindeer = train_tpe(tpe_model, tpe_predictor, x_embs, x_embs_dev, protocol, labels, n_classes, nepoch, train_batch_size, sampling_batch_size, model_saving_path, model_saving_name)

    with open(os.path.join(model_saving_path,eer_d_log_file), 'w') as f:
        f.write('min_eer: {}\n'.format(mineer))
        f.write('d_min_eer: {}'.format(mindeer))
