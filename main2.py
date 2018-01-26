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
from reportlab.lib.units import inch



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

    def _add_img_to_report(self, img_name):
        im = Image(img_name, 4 * inch, 4 * inch)
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


main()