


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


# 8. визуализировать скрытую репрезентацию
plt.imshow(encoded_imgs, cmap='gray') # там много интересных параметров в этой ф-ции, читать доки!




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


