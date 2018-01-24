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
NAME = "ZINA"

# мб стоит вычесть среднее или усилить констраст как латеральное торможение?

############################ STAGE 2 #############################################
########################## Create net ############################################
##################################################################################

# create
print("create model...")
encoding_dim = 3
encoder, decoder, autoencoder = simple_nets.create_ae_ZINA(encoding_dim=5,
                                                           input_data_shape=foveas01[0].shape,
                                                           a_koef_reg=0.0001,
                                                           koef_reg=0.0001,
                                                           activation_on_code='sigmoid',
                                                           drop_in_decoder=0.4,
                                                           drop_in_encoder=0.4)

# fit
print("fit model to data..")
plot_losses = callbacs.PlotLosses()
autoencoder.compile(optimizer='sgd', loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
history =  autoencoder.fit(foveas01, foveas01,
                epochs=3000,
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
def visualise_weights_and_biases(model, name_of_layer):
    w = model.get_layer(name_of_layer).get_weights()
    weights = w[0]
    biases = w[1]
    print ("w_shape=" + str(weights.shape) + ", b_shape=" + str(biases.shape))

    wmin = weights.min()
    wmax = weights.max()
    bmin = biases.min()
    bmax = biases.max()
    abs_max = max(abs(bmax), abs(bmin), abs(wmax), abs(wmin))
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    cax = ax1.matshow(weights, vmin=-abs_max, vmax=abs_max, cmap='bwr')
    cax = ax2.matshow([biases], vmin=-abs_max, vmax=abs_max, cmap='bwr')
    fig.colorbar(cax)
    plt.show()

visualise_weights_and_biases(decoder, NAME)

# 7. График действия ( н-1 точек, значение точки = сумма измененией (по модулю) попиксельно между соседними фреймами
def energy_of_pic_sequence(pic_sequence):
    if len(pic_sequence) < 2:
        return None
    energies = []

    #ыуммируем разницу между текущим и предыдущим
    for i in range(1, len(pic_sequence)):
        curr = pic_sequence[i]
        prev = pic_sequence[i-1]
        energies.append(utils.energy_change(prev, curr))
    everall_energy = np.array(energies).sum()
    plt.plot(range(len(pic_sequence) - 1), energies, 'g^')
    plt.title("energy of foveals path")
    plt.xlabel("number of step (n - 1)")
    plt.ylabel("avg abs energy per pixel")
    plt.ylim(0, 1.1)
    plt.show()
    return everall_energy

energy_over_path = energy_of_pic_sequence(foveas01)
print ("energy over path = " + str(energy_over_path))

# 8. визуализировать скрытую репрезентацию
plt.imshow(encoded_imgs, cmap='gray') # там много интересных параметров в этой ф-ции, читать доки!

# 9. Тест шумом
def add_noise(dataset, noise_factor, visualise=True):
    x_train_noisy = dataset + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=dataset.shape)
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    n_len = len(dataset)
    if visualise:
        plt.figure(figsize=(10, 2))
        for i in range(n_len):
            # display original
            ax = plt.subplot(2, n, i + 1)
            ax.set_title("before " + str(i))
            plt.imshow(dataset[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display with noise
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(x_train_noisy[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
    return x_train_noisy

x_train_noisy = add_noise(foveas01, noise_factor=0.01, visualise=True)

# 10. Энергия скрытого многообразия

# 11. сохраение весов и сети
def save_all(encoder, decoder, autoencoder):
    #https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
    encoder.save('encoder.h5')
    decoder.save('decoder.h5')
    autoencoder.save('autoencoder.h5')

save_all(encoder, decoder, autoencoder)

# 12. energy (entropy) per frame
#https://jamesmccaffrey.wordpress.com/2012/12/16/calculating-the-entropy-of-data-in-a-table-or-matrix/

# 13. Средний модуль веса и биаса в обученной сети
def get_abs_avg_wbiases(model, name_of_layer):
    w = model.get_layer(name_of_layer).get_weights()
    weights = w[0]
    biases = w[1]
    mw = (np.absolute(weights)).mean()
    mb = (np.absolute(biases)).mean()
    print("mean(w)=" + str(mw) + ", mean(b)=" + str(mb))

get_abs_avg_wbiases(decoder, NAME)



############################ STAGE 4 #############################################
########################## save summary ##########################################
##################################################################################

# start dialog what to save - при каких гиепрпараметрах рез-т был получен?


