import os
from pathlib import Path
import cv2
import face_recognition


def load_templates(enrollments_path):
    file_names = os.listdir(enrollments_path)
    file_names = filter(lambda x: x.endswith(('.jpg', 'png')), file_names)

    templates = dict()

    for file_name in file_names:
        person_name = Path(file_name).resolve().stem
        file_path = Path(enrollments_path, file_name)
        img = face_recognition.load_image_file(file_path)
        img_in_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(img_in_rgb)

        if len(face_locations) == 1:
            template = face_recognition.face_encodings(img_in_rgb)[0]
            templates[person_name] = template

    return templates
