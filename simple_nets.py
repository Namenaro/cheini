# -*- coding: utf-8 -*
from keras.layers import Input, Dense, Flatten, Reshape, Dropout
from keras.models import Model
import numpy as np
from keras.regularizers import l2, l1



def create_ae_ZINA(encoding_dim, input_data_shape, activation_on_code, koef_reg, a_koef_reg, drop_in_encoder, drop_in_decoder):
    summary = "с регуляризацией L1(w,b) везде," \
              " L2 на активации в коде," \
              "дропаут везде где можно :) " \
              "из инпута сразу в код, из кода сразу в выход"
    w_reg = l1(koef_reg)
    b_reg = l1(koef_reg)
    a_reg = l2(a_koef_reg)
    # Энкодер
    # Входной плейсхолдер
    input_img = Input(shape=input_data_shape)  # 28, 28, 1 - размерности строк, столбцов, фильтров одной картинки, без батч-размерности
    # Вспомогательный слой решейпинга
    flat_img = Flatten()(input_img)
    d_flat_img = Dropout(drop_in_encoder)(flat_img)
    # Кодированное полносвязным слоем представление
    encoded = Dense(encoding_dim,
                    activation=activation_on_code,
                    kernel_regularizer=w_reg,
                    bias_regularizer=b_reg,
                    activity_regularizer=a_reg,
                    name="encoder_layer")(d_flat_img)

    # Декодер
    # Раскодированное другим полносвязным слоем изображение
    input_encoded = Input(shape=(encoding_dim,))
    d_input_encoded = Dropout(drop_in_decoder)(input_encoded)
    flatten_data_len = np.prod(input_data_shape[:])
    print("flatten shape = " + str(flatten_data_len))
    flat_decoded = Dense(flatten_data_len,
                         activation='sigmoid',
                         kernel_regularizer=w_reg,
                         bias_regularizer=b_reg,
                         name="decoder_layer"
                         )(d_input_encoded)
    decoded = Reshape(input_data_shape)(flat_decoded)

    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder

def create_ae_YANA(encoding_dim, input_data_shape, activation_on_code, activity_regulariser, koef_reg, drop_in_encoder, drop_in_decoder):
    summary = "с регуляризацией L1(w,b) везде," \
              " L2 на активации в коде," \
              "дропаут везде где можно :) " \
              "из инпута сразу в код, из кода сразу в выход"
    w_reg = l1(koef_reg)
    b_reg = l1(koef_reg)
    # Энкодер
    # Входной плейсхолдер
    input_img = Input(shape=input_data_shape)  # 28, 28, 1 - размерности строк, столбцов, фильтров одной картинки, без батч-размерности
    # Вспомогательный слой решейпинга
    flat_img = Flatten()(input_img)
    d_flat_img = Dropout(drop_in_encoder)(flat_img)
    # Кодированное полносвязным слоем представление
    encoded1 = Dense(encoding_dim,
                    activation=activation_on_code,
                    kernel_regularizer=w_reg,
                    bias_regularizer=b_reg,
                    activity_regularizer=activity_regulariser,
                    name="encoder_layer1")(d_flat_img)
    encoded = Dense(encoding_dim,
                    activation=activation_on_code,
                    kernel_regularizer=w_reg,
                    bias_regularizer=b_reg,
                    activity_regularizer=activity_regulariser,
                    name="encoder_layer")(encoded1)

    # Декодер
    # Раскодированное другим полносвязным слоем изображение
    input_encoded = Input(shape=(encoding_dim,))
    d_input_encoded = Dropout(drop_in_decoder)(input_encoded)
    flatten_data_len = np.prod(input_data_shape[:])
    print("flatten shape = " + str(flatten_data_len))
    flat_decoded1 = Dense(flatten_data_len,
                         activation='sigmoid',
                         kernel_regularizer=w_reg,
                         bias_regularizer=b_reg,
                         name="decoder_layer1"
                         )(d_input_encoded)
    flat_decoded = Dense(flatten_data_len,
                         activation='sigmoid',
                         kernel_regularizer=w_reg,
                         bias_regularizer=b_reg,
                         name="decoder_layer"
                         )(flat_decoded1)
    decoded = Reshape(input_data_shape)(flat_decoded)

    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder