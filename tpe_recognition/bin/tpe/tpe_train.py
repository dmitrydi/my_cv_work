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
# возвращает hard negative пример для x_anchor - максимально близкий экземпляр из другого класса
# x_anchor - данный экземпляр определенного класса
# x_emb_W - остальные экземпляры, в т.ч. и своего класса
# tpe_predictor - предиктор модели
# labels - метки классов
# class_label - метка класса x_anchor
# random_sample_size - параметр случайной выборки экземпляров для сравнения с экземпляром x-anchor
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
# ВОЗВРАЩАЕТ
# dev_protocol - матрица Aij = 1, если i и j из одного класса, иначе - 0
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

def make_protocol(labels):
    protocol = np.matrix(np.zeros((labels.shape[0], labels.shape[0])), dtype=int)
    for label in np.unique(labels):
        label_mask = (labels == label)
        label_vect = np.matrix(np.ones(label_mask.shape)*label_mask, dtype=int)
        label_matr = label_vect.T*label_vect
        protocol += label_matr
    protocol = np.array(protocol, dtype=int)
    
    return protocol

def unpack_features(npy_archive):
# распаковка меток и фич из npy-файла
    data = np.load(npy_archive)
    labels = data[:,0].astype('int')
    embs = data[:,1:].astype('float')
    nclasses = np.unique(labels).shape[0]
    nfeatures = embs[0].shape[0]
    nexamples = embs.shape[0]
    
    return labels, embs, nclasses, nfeatures, nexamples

def train_dev_split(embs, labels, factor=0.8):
    classes = np.unique(labels)
    all_dev_inds = np.array([], dtype=int)
    all_train_inds = np.array([], dtype=int)
    for klass in classes:
        klass_indices = np.where(labels==klass)[0]
        n_examples = klass_indices.shape[0]
        if n_examples < 2:
            continue
        n_dev = max(1, int(n_examples*(1-factor)))
        n_train = n_examples - n_dev
        dev_inds = np.random.choice(klass_indices, n_dev, replace=False)
        mask = np.logical_not(np.isin(klass_indices, dev_inds))
        train_inds = np.compress(mask, klass_indices)
        all_dev_inds = np.append(all_dev_inds, dev_inds)
        all_train_inds = np.append(all_train_inds, train_inds)
    train_labels = labels[all_train_inds]
    train_embs = embs[all_train_inds]
    dev_labels = labels[all_dev_inds]
    dev_embs = embs[all_dev_inds]
            
    return train_labels, train_embs, dev_labels, dev_embs
        
        

def train_tpe(
    tpe_model, tpe_predictor, x_embs, x_embs_dev, dev_protocol, labels,
    n_classes, nepoch, batch_size, sampling_batch_size, model_saving_path, model_saving_name):
# обучает классификатор tpe_model, tpe_predictor на низкоразмерных признаках x_embs
# x_embs_dev - валидационная выборка, на которой считается loss
#
# ПАРАМЕТРЫ:
# tpe_model, tpe_predictor - классификатор, который обучаем. Инициализируется с помощью метода build_tpe
# x_embs - низкоразмерные фичи изображений, полученные на выходе нейросети
# x_embs_dev - низкоразмерные фичи development-множества
# dev_protocol - разметка для development-множества, матрица размера [N(x_embs_dev) x N(x_embs_dev}], если i-й и j-й компоненты
# относятся к одному классу, то dev_protocol[i,j] == 1, иначе - 0
# labels - метки классов для x_embs
# n_classes - кол-во классов
# nepoch - кол-во эпох обучения
# batch_size - размер батча
# sampling_batch_size - количество экземпляров x_embs для поиска максимально сложных примеров при формировании батча. Введен для того,
# чтобы не переполнять память при большом количестве экземпляров x_embs
# model_saving_path - путь для сохранения весов tpe-модели
# model_saving_name - имя для сохранения весов
#
# ВОЗВРАЩАЕТ:
# mineer - 'minimum equal error rate' - значение метрики, когда ошибка на target-попытках равна ошибке на imposter-попытках
# mindeer - параметр d [0,1], при котором ошибка модели на development-множестве равна mineer. Косинусное расстояние между фичами
# изображений в пространстве tpe-модели, которое определяет принадлежность изображений к одному или разным классам
#
# ИСПОЛЬЗОВАНИЕ:
# img1, img2 --> img1_emb, img2_emb = cnn(img1, img2) --> img1_emb_tpe, img2_emb_tpe = tpe_predictor.predict(img1_emb, img2_emb) -->
# score = img1_emb_tpe @ img2_emb_tpe.T --> если score <= mindeer, то разные классы, если score > mindeer - то одинаковые

    z = np.zeros((batch_size,))     # подаются на вход оптимизатора.
                                    #Условие "плотной границы" между положительными и негативными экземплярами a*p - a*n = 0
    mineer = 1.
    saving_name = os.path.join(model_saving_path, model_saving_name)
    nex = x_embs.shape[0]
    for _i in range(nepoch):
        print('epoch: {}'.format(_i))
        for cbatch in range(int(nex/batch_size)):
            a, p, n = get_batch(
                x_embs, tpe_predictor, labels, n_classes,
                batch_size, random_sample_size=sampling_batch_size)     # получения батча низкоразмерных фич,
                                                                        #a - anchor, p - positive, n - negative
            tpe_model.fit([a,p,n], z, batch_size=batch_size, epochs=1)  # обучение на тройках
            x_emb_dev_tpe = tpe_predictor.predict(x_embs_dev)   # предсказания для dev-множества
            tsc, isc = get_scores(x_emb_dev_tpe, dev_protocol)  # target-scores, imposter-scores
            eer, _, _, _, deer = calc_metrics(tsc, isc)         # метрики на dev-множестве
        print('EER: {:.2f}'.format(eer * 100))
        if eer < mineer:                                    # выбираем лучшую модель по параметру d (mindeer)
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
