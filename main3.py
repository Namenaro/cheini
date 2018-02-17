# -*- coding: utf-8 -*
import pandas as pd
import _pickle as pickle
import utils
import matplotlib.pyplot as plt
import seaborn as sns
#plt.xkcd() прикол!
# https://www.tutorialspoint.com/python_pandas/python_pandas_dataframe.htm

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

    # посортировать по столбцу и взять чемпионов
    champions = df.sort_values(by='wb_ratio_mean', ascending=False)[0:4]
    print (champions)
    print("the best one:")
    print(champions.iloc[0])

def exper(summaries):
    df = pd.DataFrame(summaries)
    columns = ['b_mean',
                         'w_mean',
                         'dataset',
                         'manifold_volume',
                         'loss_decrease_ratio',
                         'code_max_to_min',
                         'w_max',
                         'code_max',
                         'manifold_energy',
                         'manifold_perimeter'

                        ]
    inpordant_columns = ['b_mean',
                         'dataset',
                         'manifold_energy',
                         'manifold_perimeter'

                         ]
    some_columns = df.loc[:, inpordant_columns]

    sns.pairplot(some_columns, hue='dataset', palette='YlGn')
    plt.show()
    sns.lmplot(x="manifold_volume", y="manifold_energy", hue="dataset", data=df)
    plt.tight_layout()
    plt.show()
    champions = some_columns.sort_values(by='dataset', ascending=False)
    print(champions)


def exper2(summaries): # док-во  независоимости b_mean от кинетической и потенциальной входа
    df = pd.DataFrame(summaries)
    df['sum'] = df['kinetik energy of input'] + df['mean_poten_eneggy']
    df['dif'] = df['kinetik energy of input'] - df['mean_poten_eneggy']
    df['mul'] = df['kinetik energy of input'] * df['mean_poten_eneggy']
    df['div'] = df['kinetik energy of input'] / df['mean_poten_eneggy']
    #df['kinetik energy of input'] = df['kinetik energy of input'].apply(lambda x: round(x, ndigits=1))
    inpordant_columns = ['b_mean', 'mean_poten_eneggy', 'kinetik energy of input', 'sum','dif', 'mul', 'div']
    some_columns = df.loc[:, inpordant_columns]
    ndf = some_columns.loc[df['num_epochs'] == 5000]
    corr_mat = ndf.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_mat, vmax=1.0, vmin=-1.0, square=True, ax=ax, cmap='bwr')
    plt.show()

def exper3(summaries): # от чего зависит объем манифолда?
    df = pd.DataFrame(summaries)
    inpordant_columns = ['b_mean', 'mean_poten_eneggy', 'kinetik energy of input', 'manifold_volume', 'loss_decrease_ratio' ]
    some_columns = df.loc[:, inpordant_columns]
    ndf = some_columns.loc[df['num_epochs'] == 5000]
    corr_mat = ndf.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_mat, vmax=1.0, vmin=-1.0, square=True, ax=ax, cmap='bwr')
    plt.show()


def print_statistics(df):
    print ("STATISTICS:")
    print(df.describe())

if __name__ == "__main__":
    path2 = 'C:\\Users\\neuro\\PycharmProjects\\cheini\\SERIES\\ITOG elu i batch no reg absolutely\\summaries_dicts.pkl'
    path1 = 'C:\\Users\\neuro\\PycharmProjects\\cheini\\SERIES\\108 experiments 3 neuron\\summaries_dicts.pkl'
    summaries2 = utils.open_file(path2)
    #summaries1 = utils.open_file(path1)
    #summaries = summaries1 + summaries2
    exper(summaries2)