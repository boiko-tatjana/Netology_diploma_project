# Функция для вычисления периметра многоугольника по координатам его вершин
# (при определении состояния глаза используется вычисление периметра
# по 6 точкам глаза)

from get_distance import get_distance


def get_perimeter(eye_points):
    perimeter = 0
    for i in range(0, 6):
        segment = get_distance(eye_points[((i + 1) % 6)], eye_points[(i % 6)])
        perimeter += segment
    return perimeter
