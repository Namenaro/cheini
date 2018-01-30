# -*- coding: utf-8 -*
import pandas as pd
import _pickle as pickle
import utils
import matplotlib.pyplot as plt

# здесь мы анализируем суммари
def analize_summaries(summaries):
    df = pd.DataFrame(summaries)
    #print(df)
    # [nd] квартили, среднее, средний разброс от среднего, минимум и максимум для кадой колонки
    print_statistics(df)

    # [nd] распечатлать только некоторые столбцы:
    some_columns = df.loc[:, ['b_mean', 'w_mean', 'kinetik energy of input' ]]
    print(some_columns)

    #[2d] наглядно показать, вокруг каких значений (x, y) есть сгощение и где аулаеры
    df.plot.scatter(x='b_mean', y='w_mean', alpha=.15)
    plt.tight_layout()
    plt.show()



def print_statistics(df):
    print ("STATISTICS:")
    print(df.describe())

if __name__ == "__main__":
    path2 = 'C:\\Users\\neuro\\PycharmProjects\\cheini\\SERIES\\oximiron\\summaries_dicts.pkl'
    path = 'C:\\Users\\neuro\\PycharmProjects\\cheini\\SERIES\\real_series_1\\summaries_dicts.pkl'
    summaries = utils.open_file(path2)
    analize_summaries(summaries)