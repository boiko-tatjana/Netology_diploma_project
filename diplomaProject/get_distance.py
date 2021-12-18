# Функция для вычисления расстояния между произвольными двумя точками

import numpy as np


def get_distance(one, two):
    squared_distance = (one.x - two.x) ** 2 + (one.y - two.y) ** 2
    if squared_distance > 0:
        distance = np.sqrt(squared_distance)
    else:
        distance = 1
    return distance
