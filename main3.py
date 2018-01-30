# -*- coding: utf-8 -*
import pandas as pd
import _pickle as pickle
import utils
import matplotlib.pyplot as plt
import seaborn as sns
#plt.xkcd() прикол!

# здесь мы анализируем суммари
def analize_summaries(summaries):
    df = pd.DataFrame(summaries)
    print(df)
    # [nd] квартили, среднее, средний разброс от среднего, минимум и максимум для кадой колонки
    #print_statistics(df)

    # [nd] распечатлать только некоторые столбцы:
    some_columns = df.loc[:, ['b_mean', 'w_mean', 'kinetik energy of input' ]]
    print(some_columns)

    #[2d] наглядно показать, вокруг каких значений (x, y) есть сгощение и где аулаеры
    #df.plot.scatter(x='b_mean', y='w_mean', alpha=.15)
    sns.jointplot(x='b_mean', y='w_mean', data=df, size=5, alpha=.25,
                  color='k', marker='.')
    plt.tight_layout()
    plt.show()


    #[2d] линейная зависимость между 2 столбцами (случай с дисретной ? х). Или еще regplot
    sns.lmplot(x="kinetik energy of input", y="wb_ratio_mean", data=df)
    plt.tight_layout()
    plt.show()

    #[3d] линейная зависимость, обусловленная на 3-ю переменную (hue)
    sns.lmplot(x="loss_decrease_ratio", y="w_mean", hue="dataset", data=df)
    plt.tight_layout()
    plt.show()

    #[5d] линейная зависимость, обучловленная на третью перевенную при еще двух дискретно-значных (col, row)
    sns.lmplot(x="loss_decrease_ratio", y="w_mean", hue="dataset", col="drop_in_decoder", row="num_epochs", data=df, size=3)
    plt.tight_layout()
    plt.show()

    #g = sns.pairplot(df, hue='dataset') все со всеми, очень тяжелый и долгий отрисован
    # лучшая тулза поскать как ввлияет датасет на качественные законы взаимного распределения всех признаков!!!!
    inpordant_columns = ['b_mean', 'w_mean', 'kinetik energy of input', 'dataset']
    some_columns = df.loc[:,inpordant_columns]
    sns.pairplot(some_columns, hue='dataset')
    plt.show()

    # опарные коэффициенты корреляции между всеми столбцами. Каждая клетка - корр.коэф.между двумя переменными
    corr_mat = df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_mat, vmax=1.0, square=True, ax=ax, cmap='bwr')
    plt.show()




def print_statistics(df):
    print ("STATISTICS:")
    print(df.describe())

if __name__ == "__main__":
    path2 = 'C:\\Users\\neuro\\PycharmProjects\\cheini\\SERIES\\oximiron\\summaries_dicts.pkl'
    path = 'C:\\Users\\neuro\\PycharmProjects\\cheini\\SERIES\\real_series_1\\summaries_dicts.pkl'
    summaries = utils.open_file(path2)
    analize_summaries(summaries)