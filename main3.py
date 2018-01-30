# -*- coding: utf-8 -*
import pandas as pd
import _pickle as pickle
import utils

# здесь мы анализируем суммари
def analize_summaries(summaries):
    results_table = pd.DataFrame(summaries)
    print(results_table)

if __name__ == "__main__":
    path = 'C:\\Users\\neuro\\PycharmProjects\\cheini\\SERIES\\real_series_1\\summaries_dicts.pkl'
    summaries = utils.open_file(path)
    analize_summaries(summaries)