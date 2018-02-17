# -*- coding: utf-8 -*

import itertools
import one_experiment_report
import utils
import simple_nets
from math import floor, ceil
import matplotlib.pyplot as plt
import numpy as np
import os
import _pickle as pickle

from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras import optimizers
import time
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm



# варьируем один (или несколько) гиперпараметр - проводим таким образом серию экспериментов,
# результаты серии сводим в единый отчет: таблица из 2 столбцов (что вариьровали) и (за чем следили)
#one_experiment_report.main()

class Serial:
    def __init__(self, dataset, dataset_name='default'):
        self.batch_size = [3]
        self.code_len = [3]
        self.wb_koef_reg = [0.]
        self.num_epochs = [20]
        self.drop_in_decoder = [0.0]
        self.drop_in_encoder = [0.0]
        self.activation = ['linear']
        self.dataset = dataset
        self.dataset_name = [dataset_name]

    def _get_all_cominations(self):
        """
        :return: список словарей - всех возхможных комбинаций значений гиперпараметров
        """
        def enumdict(listed):
            myDict = {}
            for i, x in enumerate(listed):
                myDict[i] = x
            return myDict
        hypermapars_arrays = self.__dict__
        names = hypermapars_arrays.keys()
        enumerated_names = enumdict(names) # например {0: 'code_len', 1: 'activation', 2: 'num_epochs'}

        n_hyperparams = len(enumerated_names.keys())
        a = [None] * n_hyperparams
        for k in enumerated_names.keys():
            name = enumerated_names[k]
            a[k] = hypermapars_arrays[name]
        all_combinations = list(itertools.product(*a))
        all_dicts = []
        for combination in all_combinations:
            d = {}
            for i in enumerated_names.keys():
                name = enumerated_names[i]
                d[name] = combination[i]
            all_dicts.append(d)
        return all_dicts

    def make_experiments(self, folder_name=None):
        all_dicts = self._get_all_cominations()
        print("NUM EXPERIMENTS EXPECTED: " + str(len(all_dicts)))
        outer_story = []
        summaries = []
        experiment_id = 0
        if folder_name is None:
            folder_name = utils.ask_user_for_name()  # выбрать имя серии
            if folder_name is None:
                exit()
        utils.setup_folder_for_results(folder_name)
        folder_full_path = os.getcwd()
        for params in all_dicts:
            utils.setup_folder_for_results(str(experiment_id))  # имя эксперимента в серии
            e = Experiment(params)
            summary = e.run_it(outer_story=outer_story, name_of_experiment="experiment_" + str(experiment_id))
            summary['experiment_name'] = experiment_id
            all_report_line = {**params, **summary}
            summaries.append(all_report_line)
            experiment_id += 1
            os.chdir(folder_full_path)  # обратно в папку серии

        doc = SimpleDocTemplate("seria_report.pdf", pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        doc.build(outer_story)
        return summaries

from keras.regularizers import Regularizer
from keras import backend as K
class ActivityRegularizer(Regularizer):
    def __init__(self, l1=0., l2=0.):
        self.l1 = l1
        self.l2 = l2

    def __call__(self,x):
        loss = 0
        loss += self.l1 * K.sum(K.mean(K.abs(x), axis=0))
        loss += self.l2 * K.sum(K.mean(K.square(x), axis=0))
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__,
                "l1": self.l1,
                "l2": self.l2}

class Experiment:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

    def run_it(self, outer_story, name_of_experiment):
        print("RUN: " + str(self.__dict__))

        # вытаскиваем датасет из файла
        foveas01 = utils.get_dataset(self.dataset)
        a_regulariser = ActivityRegularizer(l1=0., l2=0.)

        # создаем и обучаем модельку
        en, de, ae = simple_nets.create_ae_YANA(encoding_dim=self.code_len,
                                                input_data_shape=foveas01[0].shape,
                                                activity_regulariser=a_regulariser,
                                                koef_reg=self.wb_koef_reg,
                                                activation_on_code=self.activation,
                                                drop_in_decoder=self.drop_in_decoder,
                                                drop_in_encoder=self.drop_in_encoder)

        sgd = optimizers.SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
        ae.compile(optimizer=sgd, loss='mean_squared_error')

        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
        history = ae.fit(foveas01, foveas01,
                         epochs=self.num_epochs,
                         #batch_size=ceil(len(foveas01) / 2),
                         batch_size=self.batch_size,
                         shuffle=False,
                         validation_data=(foveas01, foveas01),
                         callbacks=[early_stopping])

        # по результатам обучения на этом датасетке генерим репорт
        report = one_experiment_report.ReportOnPath(ae=ae, en=en, de=de,
                                                    dataset=foveas01,
                                                    history_obj=history,
                                                    name_of_experiment=name_of_experiment
                                                    )
        report.create_summary()
        summary, exp_outer_story = report.end()
        outer_story += exp_outer_story
        utils.save_all(encoder=en, decoder=de, autoencoder=ae)
        return summary

def make_seria_on_dataset(dataset, name_of_seria=None):
    old_dir = os.getcwd()
    utils.setup_folder_for_results("SERIES")
    s = Serial(dataset)
    summaries = s.make_experiments(folder_name=name_of_seria)
    pickle.dump(summaries, open("summaries_dicts.pkl", "wb"))
    print("summaries is saved into: " + os.getcwd())
    with open("settings.txt", "w") as text_file:
        text_file.write(str(s.__dict__))
    os.chdir(old_dir)

def get_dataset(a_dir):
    return [os.path.join(a_dir, name, 'foveas.pkl') for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


if __name__ == "__main__":
    directory = 'C:\\5x5\\test'
    dataset = get_dataset(directory)
    make_seria_on_dataset(dataset, "ITOG test 1")

