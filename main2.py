# -*- coding: utf-8 -*
import one_experiment_report

# варьируем один (или несколько) гиперпараметр - проводим таким образом серию экспериментов,
# результаты серии сводим в единый отчет: таблица из 2 столбцов (что вариьровали) и (за чем следили)
#one_experiment_report.main()

class Seria:
    def __init__(self, name, description=""):
        self.seria_folder = name
        self.description = description
