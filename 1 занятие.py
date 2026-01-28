# Библиотеки Python для научных расчетов и машинного обучения
import numpy as np

# print(np._version_)

# в Питоне динамическая типизация
x = 1
print(type(x))
x = "Hello"
print(type(x))

import sys
import array

l = [True, "2", 3.0, 4]
print([type(i) for i in l])
print(sys.getsizeof(l))

l = []
print(type(l))
print(sys.getsizeof(l))

a1 = array.array("i", [])
print(type(a1))
print(sys.getsizeof(a1))

a1 = array.array("i", [1])
print(sys.getsizeof(a1))

a1 = array.array("i", [1, 2])
print(sys.getsizeof(a1))

# NumPy и Python array - массивы хранят элементы одного типа

print(np.__version__)

# создание из списка
l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
a = np.array(l)
print(a)
print(type(a))

print("list(python)", sys.getsizeof(l))
ap = array.array("i", l)
print("array(python)", sys.getsizeof(ap))
print("array(numpy)", sys.getsizeof(a))

# "превышающее"  приведение типов
a = np.array([1.01, 2, 3, 4, 5, "a"])
print(type(a), a)

# явно задать тип
a = np.array([1.01, 2, 3, 4, 5], dtype=int)
print(type(a), a)

# одномерные массивы
a = np.array(range(2, 5))
print(a)

# многомерные массивы
a = np.array([range(i, i + 5) for i in [1, 2, 3]])
print(a)

# "из 0"
print(np.zeros(10, dtype=int))

# из 1
print(np.ones((3, 5), dtype=float))

# предопределенное значение
print(np.full((3, 3), 3.1416))

# линейная последовательность чисел
print(np.arange(0, 20, 2))

# в интервале с одинаковыми промежутками
print(np.linspace(0, 2, 5))

# равномерное распределение от 0 до 1
print(np.random.random((2, 4)))

# нормальное распределение
print(np.random.normal(0, 1, (2, 4)))

# нормальное распределение от x до y
print(np.random.randint(0, 5, (2, 2)))

# создание единичной матрицы
print(np.eye(5, dtype=int))

# Типы данных
a1 = np.zeros(10, dtype=int)
a2 = np.zeros(10, dtype="int")
a3 = np.zeros(10, dtype=np.int16)
print(a1, type(a1), a1.dtype)
print(a2, type(a2), a2.dtype)
print(a3, type(a3), a3.dtype)

# a1 = np.zeros(10, dtype = int16)
# NameError: name 'int16' is not defined

# Numerical Python = NumPy
# - атрибуты массивов
# - индексация
# - срезы
# - изменение формы
# - обьединение и разбиение

# Атрибуты: ndim - число размерностей, shape - размер каждой размерности, size - общий размер массива

np.random.seed(1)

x1 = np.random.randint(10, size=3)
print(x1)
print(x1.ndim, x1.shape, x1.size)

x2 = np.random.randint(10, size=(3, 2))
print(x2)
print(x2.ndim, x2.shape, x2.size)

x3 = np.random.randint(10, size=(3, 2, 1))
print(x3)
print(x3.ndim, x3.shape, x3.size)

# Индексация

# одномерные
a = np.array([1, 2, 3, 4, 5])
print(a[0])
print(a[-2])

a[1] = 20
print(a)

# многомерные
a = np.array([[1, 2], [3, 4]])
print(a)
print(a[0, 0])
print(a[-1, -2])
a[1, 0] = 100
print(a)

# вставки

a = np.array([1, 2, 3, 4, 5])
a[0] = 3.14
print(a.dtype)
print(a)
a.dtype = float
print(a)
print(a.dtype)

# Срезы - подмассив массива

a = np.array([1, 2, 3, 4, 5])
print(a[:3])
print(a[3:])
print(a[1:4])
print(a[::2])
print(a[1::2])

# шаг < 0
a = np.array([1, 2, 3, 4, 5])
print(a[::-1])

# Срезы в многомерном массиве
a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
print(a)
print(a[:2, :3])
print(a[:, ::2])
print(a[::-1, ::-1])
print(a[:, 0])
print(a[0, :])

# Срезы в Python - копии подмассивов, в NumPy - представления

a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
print(a)

a_2x2 = a[:2, :2]
print(a_2x2)
a_2x2[0, 0] = 999
print(a)

a_2x2 = a[:2, :2].copy()
print(a_2x2)

a_2x2[0, 0] = 1001
print(a_2x2)
print(a)

# Изменение формы массива

a = np.arange(1, 13)
print(a, a.shape, a.ndim)

a1 = a.reshape(1, 12)
print(a1, a1.shape, a1.ndim)

print(a1[0, 3])
print(a1[0, 11])

a2 = a.reshape(2, 6)
print(a2, a2.shape, a2.ndim)

a3 = a.reshape(2, 2, 3)
print(a3, a3.shape, a3.ndim)

a4 = a.reshape(1, 12, 1, 1)
print(a4, a4.shape, a4.ndim)

print(a4[0, 2, 0, 0])

a5 = a.reshape((2, 6), order="F")
print(a5, a5.shape, a5.ndim)

a = np.arange(1, 13)
print(a, a.shape, a.ndim)

print(a1[0, 3])
print(a1[0, 11])

a2 = a[np.newaxis, :]
print(a2, a2.shape, a2.ndim)
