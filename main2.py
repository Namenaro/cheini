# -*- coding: utf-8 -*

import itertools

# варьируем один (или несколько) гиперпараметр - проводим таким образом серию экспериментов,
# результаты серии сводим в единый отчет: таблица из 2 столбцов (что вариьровали) и (за чем следили)
#one_experiment_report.main()

class Serial:
    def __init__(self):
        self.code_len = None
        self.num_epochs = None
        self.activation = None

    def get_arrays(self):
        self.code_len = [1]
        self.num_epochs = [600, 700, 900]
        self.activation = ['sigmoid', 'relu']

    def get_all_cominations(self):
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

    def make_experiments(self, all_dicts):
        for params in all_dicts:
            e = Experiment(params)
            e.run_it()

class Experiment:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

    def run_it(self):
        print("RUN: " + str(self.__dict__) )

s = Serial()
s.get_arrays()
s.make_experiments(s.get_all_cominations())



