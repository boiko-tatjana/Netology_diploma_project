# Функция для определения параметров левого и правого глаза
# Для каждого глаза определяются следующие параметры:
#   - площадь многоугольника, ограничивающего глаз ("площадь глаза")
#   - периметр многоугольника, ограничивающего глаз ("периметр глаза")
#   - угол многоугольника глаза у виска ("внешний" угол глаза)
#   - угол многоугольника глаза у носа ("внутренний угол глаза")
#   - расстояние между нижним и верхним веками глаза у виска ("внешнее расстояние")
#   - расстояние между нижним и верхним веками глаза у носа ("внутреннее" расстояние)
# Для вычисления используются координаты контрольных точек лица


import os
import cv2
import dlib
import pandas as pd
from get_distance import get_distance
from get_perimeter import get_perimeter
from get_angle import get_angle
from get_eye_square import get_eye_square


def get_eyes_state(folder, state):
    directory = folder
    files = os.listdir(directory)
    os.chdir(directory)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("D:/ML/Stepik/NN_CV/models/shape_predictor_68_face_landmarks.dat")

    images_list = []

    right_eye_squares_list = []
    right_eye_perimeter_list = []
    right_eye_out_angles_list = []
    right_eye_in_angles_list = []
    right_eye_out_dist_list = []
    right_eye_in_dist_list = []

    left_eye_squares_list = []
    left_eye_perimeter_list = []
    left_eye_out_angles_list = []
    left_eye_in_angles_list = []
    left_eye_out_dist_list = []
    left_eye_in_dist_list = []

    for i in range(len(files)):

        img = cv2.imread(files[i])
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        normed_right_eye_square = None
        normed_left_eye_square = None

        right_eye_perimeter = None
        left_eye_perimeter = None

        r_out_angle = None
        r_in_angle = None
        l_out_angle = None
        l_in_angle = None

        r_in_dist = None
        r_out_dist = None
        l_in_dist = None
        l_out_dist = None

        faces = detector(frame)
        if len(faces) > 0:
            face = faces[0]
            landmarks = predictor(frame, face)

            if len(landmarks.parts()) > 0:
                # Определение параметров правого глаза
                right_eye = []
                r_all_x = 0
                r_all_y = 0
                for n in range(36, 42):
                    right_eye.append(landmarks.part(n))
                    r_all_x += landmarks.part(n).x
                    r_all_y += landmarks.part(n).y

                center_x = int(r_all_x / 6)
                center_y = int(r_all_y / 6)
                right_eye_center = {'x': center_x, 'y': center_y}

                right_eye_square = get_eye_square(right_eye, right_eye_center)

                # Определение параметров левого глаза
                left_eye = []
                l_all_x = 0
                l_all_y = 0
                for n in range(42, 48):
                    left_eye.append(landmarks.part(n))
                    l_all_x += landmarks.part(n).x
                    l_all_y += landmarks.part(n).y

                center_x = int(l_all_x / 6)
                center_y = int(l_all_y / 6)
                left_eye_center = {'x': center_x, 'y': center_y}

                left_eye_square = get_eye_square(left_eye, left_eye_center)

                # Ширина глаза (расстояние между внешним и внутренним углами)
                # используется для нормировки
                r_eye_width = get_distance(landmarks.part(36), landmarks.part(39))
                l_eye_width = get_distance(landmarks.part(42), landmarks.part(45))

                # Площадь нормируется на квадрат ширины глаза
                normed_right_eye_square = round(right_eye_square / (r_eye_width ** 2), 4)
                normed_left_eye_square = round(left_eye_square / (l_eye_width ** 2), 4)

                # Периметр нормируется на ширину глаза
                right_eye_perimeter = round(get_perimeter(right_eye) / r_eye_width, 3)
                left_eye_perimeter = round(get_perimeter(left_eye) / l_eye_width, 3)

                r_out_angle = get_angle(landmarks.part(37), landmarks.part(36), landmarks.part(41))
                r_in_angle = get_angle(landmarks.part(38), landmarks.part(39), landmarks.part(40))
                l_out_angle = get_angle(landmarks.part(44), landmarks.part(45), landmarks.part(46))
                l_in_angle = get_angle(landmarks.part(43), landmarks.part(42), landmarks.part(47))

                r_in_dist = round(get_distance(landmarks.part(38), landmarks.part(40)) / r_eye_width, 3)
                r_out_dist = round(get_distance(landmarks.part(37), landmarks.part(41)) / r_eye_width, 3)
                l_in_dist = round(get_distance(landmarks.part(43), landmarks.part(47)) / l_eye_width, 3)
                l_out_dist = round(get_distance(landmarks.part(44), landmarks.part(46)) / l_eye_width, 3)

        images_list.append(files[i])

        right_eye_squares_list.append(normed_right_eye_square)
        right_eye_perimeter_list.append(right_eye_perimeter)
        right_eye_out_angles_list.append(r_out_angle)
        right_eye_in_angles_list.append(r_in_angle)
        right_eye_out_dist_list.append(r_out_dist)
        right_eye_in_dist_list.append(r_in_dist)

        left_eye_squares_list.append(normed_left_eye_square)
        left_eye_perimeter_list.append(left_eye_perimeter)
        left_eye_out_angles_list.append(l_out_angle)
        left_eye_in_angles_list.append(l_in_angle)
        left_eye_out_dist_list.append(l_out_dist)
        left_eye_in_dist_list.append(l_in_dist)

    state = [state] * len(images_list)

    df = pd.DataFrame(list(zip(images_list,
                               right_eye_squares_list,
                               right_eye_perimeter_list,
                               right_eye_out_angles_list,
                               right_eye_in_angles_list,
                               right_eye_out_dist_list,
                               right_eye_in_dist_list,

                               left_eye_squares_list,
                               left_eye_perimeter_list,
                               left_eye_out_angles_list,
                               left_eye_in_angles_list,
                               left_eye_out_dist_list,
                               left_eye_in_dist_list,

                               state)),
                      columns=['image',
                               'r_square', 'r_perimeter', 'r_out_angle', 'r_in_angle', 'r_out_dist', 'r_in_dist',
                               'l_square', 'l_perimeter', 'l_out_angle', 'l_in_angle', 'l_out_dist', 'l_in_dist',
                               'state'])
    return df
