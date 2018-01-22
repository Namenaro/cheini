# -*- coding: utf-8 -*
from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model
import numpy as np


def create_ae_MASHA(encoding_dim, input_data_shape):
    # Энкодер
    # Входной плейсхолдер
    input_img = Input(shape=input_data_shape)  # 28, 28, 1 - размерности строк, столбцов, фильтров одной картинки, без батч-размерности
    # Вспомогательный слой решейпинга
    flat_img = Flatten()(input_img)
    # Кодированное полносвязным слоем представление
    encoded = Dense(encoding_dim, activation='relu')(flat_img)

    # Декодер
    # Раскодированное другим полносвязным слоем изображение
    input_encoded = Input(shape=(encoding_dim,))
    flatten_data_len = np.prod(input_data_shape[:])
    print ("flatten shape = " + str(flatten_data_len))
    flat_decoded = Dense(flatten_data_len, activation='sigmoid')(input_encoded)
    decoded = Reshape(input_data_shape)(flat_decoded)

    # Модели, в конструктор первым аргументом передаются входные слои, а вторым выходные слои
    # Другие модели можно так же использовать как и слои
    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder

if __name__ == '__main__':
    pass