# -*- coding: utf-8 -*
print ("MAIN_FIT")
############################ STAGE 1 #############################################
###################### Prerocess data ############################################
##################################################################################
import utils
import simple_nets
from math import floor, ceil

# вытаскиваем датасет из файла
foveas01, points = utils.get_dataset(READ_DAMMY=True)

# мб стоит вычесть среднее или усилить констраст как латеральное торможение?

############################ STAGE 2 #############################################
########################## Create net ############################################
##################################################################################

# create
print("create model...")
encoder, decoder, autoencoder = simple_nets.create_ae_MASHA(encoding_dim=5, input_data_shape=foveas01[0].shape)

# fit
print("fit model to data..")
autoencoder.compile(optimizer='sgd', loss='mean_squared_error')
autoencoder.fit(foveas01, foveas01,
                epochs=1400,
                batch_size=ceil(len(foveas01)/2),
                shuffle=True,
                validation_data=(foveas01, foveas01))

############################ STAGE 3 #############################################
########################## evaluate res ##########################################
##################################################################################

# visualise reconstruction
# visualise manifold
# weights / biases  распределения
# energy of temporal changes
# energy (entropy) per frame


############################ STAGE 4 #############################################
########################## save summary ##########################################
##################################################################################

# start dialog what to save


