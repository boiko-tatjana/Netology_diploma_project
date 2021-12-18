# Функция для вычисления угла в градусах по координатам трех точек

import numpy as np
from get_distance import get_distance


def get_angle(point_a, point_c, point_b):
    ca_cb = (point_a.x - point_c.x) * (point_b.x - point_c.x) + \
            (point_a.y - point_c.y) * (point_b.y - point_c.y)
    abs_ca = get_distance(point_a, point_c)
    abs_cb = get_distance(point_b, point_c)
    cos_c = ca_cb / (abs_ca * abs_cb)

    if cos_c < (-1):
        cos_c = -1
    if cos_c > 1:
        cos_c = 1

    angle = round(np.arccos(cos_c) / np.pi * 180, 1)
    assert isinstance(angle, object)
    return angle
