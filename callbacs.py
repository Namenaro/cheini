


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
            plt.imshow(dataset[i], cmap='gray', vmax=1.0, vmin=0.0)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display with noise
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(x_train_noisy[i], cmap='gray', vmax=1.0, vmin=0.0)
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


