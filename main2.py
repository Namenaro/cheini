# -*- coding: utf-8 -*

import utils
import simple_nets
from math import floor, ceil
import matplotlib.pyplot as plt
import numpy as np
import callbacs
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard

# выбрать папку для сохранения результатов
############################ STAGE 1 #############################################
###################### Prerocess data ############################################
##################################################################################
# вытаскиваем датасет из файла
foveas01, points = utils.get_dataset(READ_DAMMY=True)

# мб стоит вычесть среднее или усилить констраст как латеральное торможение?

############################ STAGE 2 #############################################
########################## Create net ############################################
##################################################################################

# create
print("create model...")
encoding_dim = 3
encoder, decoder, autoencoder = simple_nets.create_ae_MASHA(encoding_dim=3, input_data_shape=foveas01[0].shape)

# fit
print("fit model to data..")
plot_losses = callbacs.PlotLosses()
autoencoder.compile(optimizer='sgd', loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
history =  autoencoder.fit(foveas01, foveas01,
                epochs=10000,
                batch_size=ceil(len(foveas01)/2),
                shuffle=True,
                validation_data=(foveas01, foveas01),
                callbacks=[early_stopping])

############################ STAGE 3 #############################################
########################## evaluate res ##########################################
##################################################################################

# 1. График сходимости
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# 2. Конечная и начальная ошибки
initial_loss = history.history['loss'][0]
end_loss = history.history['loss'][-1]
print ("initial_loss=" + str(initial_loss) + ", end_loss=" + str(end_loss))

# 3. Визуализировать реконструкцию на батче
encoded_imgs = encoder.predict(foveas01)
decoded_imgs = decoder.predict(encoded_imgs)

n = len(foveas01)
plt.figure(figsize=(10, 2))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    ax.set_title( "original " + str(i))
    plt.imshow(foveas01[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# 4. Визаулизировать многообразие: точками в СО многообразия
encoded_imgs = encoder.predict(foveas01)
def visualise_points_manifold(encoded_images, i=0, j=1):
    plt.figure(figsize=(6, 6))
    plt.scatter(encoded_images[:, i], encoded_images[:, j], c=range(0, len(encoded_images)))
    plt.colorbar()
    plt.show()

visualise_points_manifold(encoded_imgs, 0, 1)

# 5. Визуализировать многообразие: картинками, пробегая по решетке кодов 2д
# display a 2D manifold of the digits

def visualise_points_manifold(imgs, i=0, j=1):
    encoded_imgs = encoder.predict(imgs)
    grid_n = 15
    pic_side = imgs[0].shape[0] # картинки типа всегда квадратыне
    figure = np.zeros((pic_side * grid_n, pic_side * grid_n))

    grid_x = np.linspace(2*min(encoded_imgs[:, i]), 2*max(encoded_imgs[:, i]), grid_n)
    grid_y = np.linspace(2*min(encoded_imgs[:, j]), 2*max(encoded_imgs[:, j]), grid_n)


    for ix, xi in enumerate(grid_x):
        for iy, yi in enumerate(grid_y):
            code = encoded_imgs[0]
            code[i] = xi
            code[j] = yi
            decoded_images = decoder.predict(np.array([code]), batch_size=1)
            figure[ix * pic_side: (ix + 1) * pic_side,
                   iy * pic_side: (iy + 1) * pic_side] = decoded_images[0]

    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.show()


visualise_points_manifold(foveas01, 0, 1)

# 6. Визуализация весов как картинки


# 7. График действия ( н-1 точек, значение точки = сумма измененией (по модулю) попиксельно между соседними фреймами


# energy (entropy) per frame


############################ STAGE 4 #############################################
########################## save summary ##########################################
##################################################################################

# start dialog what to save - при каких гиепрпараметрах рез-т был получен?


