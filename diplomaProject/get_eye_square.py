# Функция для вычисления площади шестиугольника, ограничивающего глаз (левый или правый).
# В функции используется разделение шестиугольника на 6 треугольников, т.е.
# реализован метод триангуляции

import numpy as np


def get_eye_square(eye_points, eye_center):
    squares_sum = 0
    for i in range(0, 6):
        point_a = eye_points[(i % 6)]
        point_b = eye_points[((i + 1) % 6)]
        point_c = eye_center

        a = np.sqrt((point_b.x - point_c['x']) ** 2 + (point_b.y - point_c['y']) ** 2)
        b = np.sqrt((point_a.x - point_c['x']) ** 2 + (point_a.y - point_c['y']) ** 2)
        c = np.sqrt((point_a.x - point_b.x) ** 2 + (point_a.y - point_b.y) ** 2)

        # Вычисление полупериметра
        p = (a + b + c) / 2.0

        # Для вычисления площади треугольника используется формула Герона
        if p > a and p > b and p > c:
            s = np.sqrt(p * (p - a) * (p - b) * (p - c))
        else:
            s = 0
        squares_sum += s
    return squares_sum
