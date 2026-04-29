import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2))


# x = np.linspace(-6, 6, 30)
# y = np.linspace(-10, 10, 50)


# print(X.shape)
# print(Y.shape)

# print(X)
# print(Y)

# Z = f(X, Y)

# fig = plt.figure()
# ax = plt.axes()

# ax.scatter3D(X, Y, Z, c=Z)
# ax.plot_wireframe(X, Y, Z)
# ax.plot_surface(X, Y, Z, cmap='virdis')

# angle = np.linspace(0, 2 * np.pi, 50)
# r = np.linspace(0, 6, 30)

# R, Angle = np.meshgrid(r, angle)

# X = r * np.sin(Angle)
# Y = r * np.cos(Angle)

# Z = f(X, Y)
# ax.plot_surface(X, Y, Z, cmap='virdis')

# angle = 1.5 * np.pi * np.random.random(50)
# r = np.linspace(0, 6, 1000)

# R, Angle = np.meshgrid(r, angle)

# X = R * np.sin(angle)
# Y = R * np.cos(angle)
# Z = f(X, Y)

# x = r * np.sin(angle)
# y = r * np.cos(angle)
# z = f(x, y)

# ax.plot_surface(X, Y, Z, cmap='virdis')
# ax.plot_trisurf(x, y, z, cmap='virdis')

# plt.show()

# Seaborn

import seaborn as sns

sns.set_style("darkgrid")

df = pd.read_csv("cars.csv")

print(cars.head())

## Числовые данные
## парная

# sns.pairplot(cars)

# sns.pairplot(data=cars, hue='transmission')

## Тепловая карта

# cars_corr = cars[['year', 'selling_price', 'seats']]

# sns.heatmap(cars_corr, cmap='virdis', annot=True)

## Д.рассеивания

# sns.scatterplot(x='seats', y='mileage', data=cars, hue='fuel')
# sns.scatterplot(x='year', y='mselling_price', data=cars)

## Д.рассеивания + лин. регрессия

# sns.regplot(x='seats', y='mileage', data=cars)

# sns.regplot(x='seats', y='mileage', data=cars, hue='fuel')

# sns.regplot(x='seats', y='mileage', data=cars, kind='scatter', hue='fuel')

# sns.regplot(x='seats', y='mileage', data=cars, kind='scatter', col='fuel', col_wrap=2, hue='transmission')

# sns.regplot(x='seats', y='mileage', data=cars, kind='scatter', col='transmission', col_wrap=2, hue='fuel')

## Д.рассеивания + лин. регрессия

# sns.lmplot(x='seats', y='mileage', data=cars, col='transmission', col_wrap=2, hue='fuel')

# Линейный график

# sns.lineplot(x='seats', y='mileage', data=cars, hue='fuel')

# Сводная диаграмма

# sns.jointplot(x = 'year', y='selling_price', data=cars, kind='kde')

# sns.jointplot(x = 'year', y='selling_price', data=cars, kind='hex')

sns.jointplot(x="year", y="selling_price", data=cars, hue="transmission")

## Категории и числа

# sns.barplot(x = 'fuel', y='selling_price', data=cars, estimator=np.mean)

# sns.barplot(x = 'fuel', y='selling_price', data=cars, estimator=np.mean, hue='transmission')

# sns.catplot(x = 'fuel', y='selling_price', data=cars, estimator=np.mean, kind='bar', col='selling_price')

# sns.pointplot(x = 'fuel', y='selling_price', data=cars, estimator=np.mean, hue='transmission')

# sns.boxplot(x='fuel', y='selling_price', data=cars, hue='transmission')

# sns.viplinplot(x='fuel', y='selling_price', data=cars, hue='transmission')

# sns.stripplot(x='fuel', y='selling_price', data=cars, hue='transmission')

# sns.viplinplot(x='fuel', y='selling_price', data=cars)

# sns.stripplot(x='fuel', y='selling_price', data=cars)

g = sns.catplot(x="fuel", y="selling_price", data=cars, kind="box")

sns.stripplot(x="fuel", y="selling_price", data=cars, ax=g.ax)

plt.show()
