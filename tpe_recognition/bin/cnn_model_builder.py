from keras.layers import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import PReLU
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow as tf
import os
from keras.models import Sequential
from keras import backend as K
K.set_image_dim_ordering('th')

# building model and compiling

def build_cnn(h_img, w_img, n_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(3, h_img, w_img), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def train_save_cnn(
    model, train_path=None, test_path=None, saving_path='', saving_name='model.h5',
    img_height=120, img_width=60, nepoch=50, steps_per_epoch=30, n_samples_train=3000,
    n_samples_test=600, batch_size=100, **imDataGen_kwargs):
    
    train_datagen = ImageDataGenerator(**imDataGen_kwargs)
    test_datagen = ImageDataGenerator(rescale=imDataGen_kwargs['rescale'])
    
    train_generator=train_datagen.flow_from_directory(
        train_path, target_size=(img_height,img_width), batch_size=batch_size)
    validation_generator = test_datagen.flow_from_directory(
        test_path, target_size=(img_height,img_width), batch_size=batch_size)
    
    model.fit_generator(train_generator,
                       steps_per_epoch=steps_per_epoch,
                       epochs=nepoch,
                       validation_data=validation_generator,
                       validation_steps=n_samples_test//batch_size+int(bool(n_samples_test%batch_size)))
    
    model_name=os.path.join(saving_path,saving_name)
    model.save(model_name)