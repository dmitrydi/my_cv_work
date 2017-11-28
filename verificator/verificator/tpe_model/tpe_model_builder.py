from keras.layers import Dense, Lambda, Input, merge, Reshape, Flatten
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import SGD
import keras.backend as K
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from .tpe_metrics import get_scores, calc_metrics
from keras import backend as K
K.set_image_dim_ordering('th')

def triplet_loss(y_true, y_pred):
    return -K.mean(K.log(K.sigmoid(y_pred)))

def triplet_merge(inputs):
    a, p, n = inputs

    return K.sum(a * (p - n), axis=1)

def triplet_merge_shape(input_shapes):
    return (input_shapes[0][0], 1)

def build_tpe(n_in, n_out, W_pca=None):
# строит две модели в соответсвии с подходом TPE:
# первая модель принимает на вход массив из троек embedding-векторов [a, p, n], a,p,n = [?, n_in]
# и выдает массив e = [?,n_out] a*(p-n) - разницу косинусных расстояний между (a,p) и (a,n), a={anchor}, p={positive}, n={negative}
# вторая модель принимает embedding-вектор a и выдает W_pca*a (нормированный)
# PARAMS:
# n_in - размерность входных векторов
# n_out - размерность выходных векторов W_pca*a
# W_pca - матрица весов моделей W_pca = [n_in, n_out]
    a = Input(shape=(n_in,))
    p = Input(shape=(n_in,))
    n = Input(shape=(n_in,))

    if W_pca is None:
        W_pca = np.ones((n_in, n_out))
        
    assert W_pca.shape == (n_in, n_out)

    # умножение W_pca * x с последующей нормализацией
    base_model = Sequential()
    base_model.add(Dense(n_out, input_dim=n_in, bias=False, weights=[W_pca], activation='linear'))
    base_model.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))
    a_emb = base_model(a)
    p_emb = base_model(p)
    n_emb = base_model(n)
    #e = layers.Dot(axes=1)([a_emb,layers.subtract([p_emb,n_emb])])
    e = merge([a_emb, p_emb, n_emb], mode=triplet_merge, output_shape=triplet_merge_shape)
    model = Model(input=[a, p, n], output=e)
    predict = Model(input=a, output=a_emb)
    model.compile(loss=triplet_loss, optimizer='rmsprop')

    return model, predict
