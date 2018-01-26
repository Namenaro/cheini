# -*- coding: utf-8 -*

import utils
import simple_nets
from math import floor, ceil
import matplotlib.pyplot as plt
import numpy as np

from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras import optimizers
import time
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm



def main():
    name = utils.ask_user_for_name()  # выбрать папку для сохранения результатов
    # вытаскиваем датасет из файла
    foveas01, points = utils.get_dataset(READ_DAMMY=True)

    #создаем и обучаем модельку
    encoding_dim = 2
    print("create model...")
    en, de, ae = simple_nets.create_ae_ZINA(encoding_dim=encoding_dim,
                                            input_data_shape=foveas01[0].shape,
                                            a_koef_reg=0.001,
                                            koef_reg=0.0001,
                                            activation_on_code='sigmoid',
                                            drop_in_decoder=0.1,
                                            drop_in_encoder=0.1)

    # fit
    print("fit model to data..")
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    ae.compile(optimizer=sgd, loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
    history = ae.fit(foveas01, foveas01,
                              epochs=200,
                              batch_size=ceil(len(foveas01) / 2),
                              shuffle=True,
                              validation_data=(foveas01, foveas01),
                              callbacks=[early_stopping])

    # по результатам обучения на этом датасетке генерим репорт
    report = ReportOnPath(ae=ae, en=en, de=de, dataset=foveas01, experiment_name=name, history_obj=history)
    report.setup()
    report.create_summary()
    summary = report.end()
    print (summary)
    utils.save_all(encoder=en, decoder=de, autoencoder=ae)

class ReportOnPath:
    def __init__(self, ae, en, de, history_obj, dataset, experiment_name, SHOW=False):
        self.autoencoder = ae
        self.history_obj = history_obj
        self.encoder = en
        self.decoder = de
        self.dataset = dataset
        self.exp_name = experiment_name
        self.story = []
        self.summary = {}
        self.SHOW = SHOW

    def setup(self):
        utils.setup_folder_for_results(self.exp_name)

    def create_summary(self):
        self.report_loss_decrease()
        self.visualise_reconstruction()
        self.visualise_manifold_2d(i=0, j=1)
        self.kinetik_energy_of_input_sequence()
        self.code_visualise()
        self.analise_encoder_params()
        self.analise_decoder_params()

    def report_loss_decrease(self):
        # График сходимости
        plt.plot(self.history_obj.history['loss'])
        plt.plot(self.history_obj.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        self._savefig("loss.png")
        self._add_img_to_report("loss.png")

        # Конечная и начальная ошибки
        initial_loss = self.history_obj.history['val_loss'][0]
        end_loss = self.history_obj.history['val_loss'][-1]
        report_str = "initial_loss=" + str(initial_loss) + ", end_loss=" + str(end_loss)
        self._add_text_to_report(report_str)
        self._add_to_summary('loss_decrease_ratio', initial_loss/end_loss)

    def visualise_reconstruction(self):
        #Визуализировать реконструкцию на батче
        encoded_imgs = self.encoder.predict(self.dataset)
        decoded_imgs = self.decoder.predict(encoded_imgs)

        n = len(self.dataset)
        plt.figure(figsize=(10, 2))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            ax.set_title("original " + str(i))
            plt.imshow(self.dataset[i], cmap='gray', vmax=1.0, vmin=0.0)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i], cmap='gray', vmax=1.0, vmin=0.0)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        self._savefig("reconstruction.png")
        self._add_img_to_report("reconstruction.png")

    def visualise_manifold_2d(self, i, j):
        self._add_text_to_report("visualise manifold by " + str(i) + ", " + str(j))
        # Визаулизировать многообразие: точками в СО многообразия
        encoded_imgs = self.encoder.predict(self.dataset)
        code_len = len(encoded_imgs[0])
        if i >= code_len or j >= code_len:
            return None
        plt.figure(figsize=(6, 6))
        plt.scatter(encoded_imgs[:, i], encoded_imgs[:, j], c=range(0, len(encoded_imgs)))
        plt.colorbar()
        filename=str(i) + "_" + str(j) +"_manifold(as points).png"
        self._savefig(filename)
        self._add_img_to_report(filename)

        # Визуализировать многообразие: картинками, пробегая по решетке кодов 2д
        grid_n = 15
        pic_side = self.dataset[0].shape[0]  # картинки типа всегда квадратыне
        figure = np.zeros((pic_side * grid_n, pic_side * grid_n))

        grid_x = np.linspace(2 * min(encoded_imgs[:, i]), 2 * max(encoded_imgs[:, i]), grid_n)
        grid_y = np.linspace(2 * min(encoded_imgs[:, j]), 2 * max(encoded_imgs[:, j]), grid_n)

        for ix, xi in enumerate(grid_x):
            for iy, yi in enumerate(grid_y):
                code = encoded_imgs[0]
                code[i] = xi
                code[j] = yi
                decoded_images = self.decoder.predict(np.array([code]), batch_size=1)
                figure[ix * pic_side: (ix + 1) * pic_side,
                iy * pic_side: (iy + 1) * pic_side] = decoded_images[0]

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='gray', vmax=1.0, vmin=0.0)
        filename2 = str(i) + "_" + str(j) + "_manifold(as pictures).png"
        self._savefig(filename2)
        self._add_text_to_report(filename2)

    def kinetik_energy_of_input_sequence(self):
        energies = self._energy_of_sequence(self.dataset)
        overall_energy = np.array(energies).sum()
        plt.plot(range(len(self.dataset) - 1), energies, 'g^')
        plt.title("energy of foveals path")
        plt.xlabel("number of step (n - 1)")
        plt.ylabel("avg abs energy per 1 (!!!!!) pixel")
        plt.ylim(0, 1.1)
        filename = 'energy of foveals path.png'
        self._savefig(filename)
        self._add_text_to_report("energy of INPUT sequence")
        self._add_img_to_report(filename)
        self._add_to_summary('kinetik energy of input', overall_energy)
        self._add_text_to_report('kinetik energy of input = ' + str(overall_energy))

    def code_visualise(self):
        encoded_imgs = self.encoder.predict(self.dataset)
        plt.imshow(encoded_imgs, cmap='gray')
        filename = "hidden codes.png"
        self._savefig(filename)
        self._add_img_to_report(filename)

        mmean_activation = (np.absolute(encoded_imgs)).mean()
        max_activation = (np.absolute(encoded_imgs)).max()
        max_to_mean = max_activation/mmean_activation
        self._add_text_to_report("codes -> mean:"+str(mmean_activation) + ", max/mean=" + str(max_to_mean))
        self._add_to_summary('code_mean', mmean_activation)
        self._add_to_summary('code_max', max_activation)
        self._add_to_summary('code_max_to_min', max_to_mean)



    def _energy_of_sequence(self, pic_sequence):
        if len(pic_sequence) < 2:
            return 0
        energies = []
        # ыуммируем разницу между текущим и предыдущим
        for i in range(1, len(pic_sequence)):
            curr = pic_sequence[i]
            prev = pic_sequence[i - 1]
            energies.append(utils.energy_change(prev, curr))
        return energies

    def analise_encoder_params(self):
        self._analise_layer(self.encoder, 'encoder_layer')

    def analise_decoder_params(self):
        self._analise_layer(self.decoder, 'decoder_layer')

    def _analise_layer(self, keras_model, layer_name):
        w = keras_model.get_layer(layer_name).get_weights()
        weights = w[0]
        biases = w[1]
        print("w_shape=" + str(weights.shape) + ", b_shape=" + str(biases.shape))

        wmin = weights.min()
        wmax = weights.max()
        bmin = biases.min()
        bmax = biases.max()
        abs_max = max(abs(bmax), abs(bmin), abs(wmax), abs(wmin))
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        cax = ax1.matshow(weights, vmin=-abs_max, vmax=abs_max, cmap='bwr')
        cax = ax2.matshow([biases], vmin=-abs_max, vmax=abs_max, cmap='bwr')
        fig.colorbar(cax)
        filename = "FIRST (encoder) layer.png"
        self._savefig(filename)
        self._add_img_to_report(filename)

        mean_w = (np.absolute(weights)).mean()
        max_w = (np.absolute(weights)).max()
        w_max_to_mean = max_w / mean_w
        self._add_text_to_report(layer_name + " Weights -> mean:" + str(mean_w) + ", max/mean=" + str(w_max_to_mean))
        self._add_to_summary('w_mean', mean_w)
        self._add_to_summary('w_max', max_w)
        self._add_to_summary('w_max_to_min', w_max_to_mean)

        mean_b = (np.absolute(biases)).mean()
        max_b = (np.absolute(biases)).max()
        b_max_to_mean = max_b /mean_b
        self._add_text_to_report(layer_name + " Biases -> mean:" + str(mean_b) + ", max/mean=" + str(b_max_to_mean))
        self._add_to_summary('wb_ratio_mean', mean_w/mean_b)
        self._add_to_summary('b_mean', mean_b)

    def _add_img_to_report(self, img_name):
        im = utils.get_image_for_report(img_name, width=8*cm)
        self.story.append(im)

    def _savefig(self, filename):
        if self.SHOW:
            plt.show()
        plt.savefig(filename)
        plt.close()

    def _add_text_to_report(self, text):
        ptext = '<font size=12>%s</font>' % text
        styles = getSampleStyleSheet()
        self.story.append(Paragraph(ptext, styles["Normal"]))

    def _add_to_summary(self, key, val):
        self.summary[key] = val


    def end(self):
        doc = SimpleDocTemplate("report.pdf", pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        doc.build(self.story)
        return self.summary

##################################################################################


#main()