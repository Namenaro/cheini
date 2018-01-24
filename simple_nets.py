# -*- coding: utf-8 -*
from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model
import numpy as np
from keras.regularizers import l2, l1


def create_ae_MASHA(encoding_dim, input_data_shape, activation_on_code):
    summary = "без регуляризации, из инпута сразу в код, из кода сразу в выход "
    name_of_last_layer = 'MASHA'
    # Энкодер
    # Входной плейсхолдер
    input_img = Input(shape=input_data_shape)  # 28, 28, 1 - размерности строк, столбцов, фильтров одной картинки, без батч-размерности
    # Вспомогательный слой решейпинга
    flat_img = Flatten()(input_img)
    # Кодированное полносвязным слоем представление
    encoded = Dense(encoding_dim, activation=activation_on_code)(flat_img)

    # Декодер
    # Раскодированное другим полносвязным слоем изображение
    input_encoded = Input(shape=(encoding_dim,))
    flatten_data_len = np.prod(input_data_shape[:])
    print("flatten shape = " + str(flatten_data_len))
    flat_decoded = Dense(flatten_data_len, activation='sigmoid', name=name_of_last_layer)(input_encoded)
    decoded = Reshape(input_data_shape)(flat_decoded)

    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder

def create_ae_ANYA(encoding_dim, input_data_shape, activation_on_code, koef_reg):
    summary = "с регуляризацией L2(w,b) только на среднем слое, из инпута сразу в код, из кода сразу в выход"
    w_reg = l2(koef_reg)
    b_reg = l2(koef_reg)
    # Энкодер
    # Входной плейсхолдер
    input_img = Input(shape=input_data_shape)  # 28, 28, 1 - размерности строк, столбцов, фильтров одной картинки, без батч-размерности
    # Вспомогательный слой решейпинга
    flat_img = Flatten()(input_img)
    # Кодированное полносвязным слоем представление
    encoded = Dense(encoding_dim,
                    activation=activation_on_code,
                    kernel_regularizer=w_reg,
                    bias_regularizer=b_reg)(flat_img)

    # Декодер
    # Раскодированное другим полносвязным слоем изображение
    input_encoded = Input(shape=(encoding_dim,))
    flatten_data_len = np.prod(input_data_shape[:])
    print("flatten shape = " + str(flatten_data_len))
    flat_decoded = Dense(flatten_data_len, activation='sigmoid', name='ANYA')(input_encoded)
    decoded = Reshape(input_data_shape)(flat_decoded)

    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder

def create_ae_MILA(encoding_dim, input_data_shape, activation_on_code, koef_reg):
    summary = "с регуляризацией L2(w,b) везде, из инпута сразу в код, из кода сразу в выход"
    w_reg = l2(koef_reg)
    b_reg = l2(koef_reg)
    # Энкодер
    # Входной плейсхолдер
    input_img = Input(shape=input_data_shape)  # 28, 28, 1 - размерности строк, столбцов, фильтров одной картинки, без батч-размерности
    # Вспомогательный слой решейпинга
    flat_img = Flatten()(input_img)
    # Кодированное полносвязным слоем представление
    encoded = Dense(encoding_dim,
                    activation=activation_on_code,
                    kernel_regularizer=w_reg,
                    bias_regularizer=b_reg)(flat_img)

    # Декодер
    # Раскодированное другим полносвязным слоем изображение
    input_encoded = Input(shape=(encoding_dim,))
    flatten_data_len = np.prod(input_data_shape[:])
    print("flatten shape = " + str(flatten_data_len))
    flat_decoded = Dense(flatten_data_len,
                         activation='sigmoid',
                         name='MILA',
                         kernel_regularizer=w_reg,
                         bias_regularizer=b_reg
                         )(input_encoded)
    decoded = Reshape(input_data_shape)(flat_decoded)

    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder


def create_ae_IRA(encoding_dim, input_data_shape, activation_on_code, koef_reg):
    summary = "с регуляризацией L1(w,b) везде, из инпута сразу в код, из кода сразу в выход"
    w_reg = l1(koef_reg)
    b_reg = l1(koef_reg)
    # Энкодер
    # Входной плейсхолдер
    input_img = Input(shape=input_data_shape)  # 28, 28, 1 - размерности строк, столбцов, фильтров одной картинки, без батч-размерности
    # Вспомогательный слой решейпинга
    flat_img = Flatten()(input_img)
    # Кодированное полносвязным слоем представление
    encoded = Dense(encoding_dim,
                    activation=activation_on_code,
                    kernel_regularizer=w_reg,
                    bias_regularizer=b_reg)(flat_img)

    # Декодер
    # Раскодированное другим полносвязным слоем изображение
    input_encoded = Input(shape=(encoding_dim,))
    flatten_data_len = np.prod(input_data_shape[:])
    print("flatten shape = " + str(flatten_data_len))
    flat_decoded = Dense(flatten_data_len,
                         activation='sigmoid',
                         name='IRA',
                         kernel_regularizer=w_reg,
                         bias_regularizer=b_reg
                         )(input_encoded)
    decoded = Reshape(input_data_shape)(flat_decoded)

    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder

def create_ae_YANA(encoding_dim, input_data_shape, activation_on_code, koef_reg, a_koef_reg):
    summary = "с регуляризацией L1(w,b) везде, L2 на активации в коде, из инпута сразу в код, из кода сразу в выход"
    w_reg = l1(koef_reg)
    b_reg = l1(koef_reg)
    a_reg = l2(a_koef_reg)
    # Энкодер
    # Входной плейсхолдер
    input_img = Input(shape=input_data_shape)  # 28, 28, 1 - размерности строк, столбцов, фильтров одной картинки, без батч-размерности
    # Вспомогательный слой решейпинга
    flat_img = Flatten()(input_img)
    # Кодированное полносвязным слоем представление
    encoded = Dense(encoding_dim,
                    activation=activation_on_code,
                    kernel_regularizer=w_reg,
                    bias_regularizer=b_reg,
                    activity_regularizer=a_reg)(flat_img)

    # Декодер
    # Раскодированное другим полносвязным слоем изображение
    input_encoded = Input(shape=(encoding_dim,))
    flatten_data_len = np.prod(input_data_shape[:])
    print("flatten shape = " + str(flatten_data_len))
    flat_decoded = Dense(flatten_data_len,
                         activation='sigmoid',
                         name='YANA',
                         kernel_regularizer=w_reg,
                         bias_regularizer=b_reg
                         )(input_encoded)
    decoded = Reshape(input_data_shape)(flat_decoded)

    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder

def create_ae_ZINA(encoding_dim, input_data_shape, activation_on_code, koef_reg, a_koef_reg):
    summary = "с регуляризацией L1(w,b) везде, L2 на активации в коде, из инпута сразу в код, из кода сразу в выход"
    w_reg = l1(koef_reg)
    b_reg = l1(koef_reg)
    a_reg = l2(a_koef_reg)
    # Энкодер
    # Входной плейсхолдер
    input_img = Input(shape=input_data_shape)  # 28, 28, 1 - размерности строк, столбцов, фильтров одной картинки, без батч-размерности
    # Вспомогательный слой решейпинга
    flat_img = Flatten()(input_img)
    # Кодированное полносвязным слоем представление
    encoded = Dense(encoding_dim,
                    activation=activation_on_code,
                    kernel_regularizer=w_reg,
                    bias_regularizer=b_reg,
                    activity_regularizer=a_reg)(flat_img)

    # Декодер
    # Раскодированное другим полносвязным слоем изображение
    input_encoded = Input(shape=(encoding_dim,))
    flatten_data_len = np.prod(input_data_shape[:])
    print("flatten shape = " + str(flatten_data_len))
    flat_decoded = Dense(flatten_data_len,
                         activation='sigmoid',
                         name='YANA',
                         kernel_regularizer=w_reg,
                         bias_regularizer=b_reg
                         )(input_encoded)
    decoded = Reshape(input_data_shape)(flat_decoded)

    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder