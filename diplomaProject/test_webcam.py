import cv2
import dlib
import face_recognition
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms as transforms
from PIL import Image
from statistics import mode
import datetime
import pickle
from collections import Counter

from get_eye_square import get_eye_square
from get_distance import get_distance
from get_perimeter import get_perimeter
from get_angle import get_angle

from load_templates import load_templates

# Path to the file with pretrained random forest classifier for shape features
random_forest_model = pickle.load(open("./models/shape_random_forest.pkl", "rb"))

# Path to the file with pretrained random forest classifier for texture features
texture_random_forest_model = pickle.load(open("./models/texture_random_forest_2021_05_16.pkl", "rb"))

# Path to the file with template for 68 facial landmarks
predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")

# Neural network class definition
train_on_gpu = torch.cuda.is_available()


# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 5)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32 * 53 * 53, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 32 * 53 * 53)
        x = f.relu(self.fc1(x))
        x = self.dropout(f.relu(self.fc2(x)))
        x = self.softmax(self.fc3(x))
        return x


# create a complete CNN
neural_network_model = Net()

# Path to the file with weights for pretrained neural network model
neural_network_model.load_state_dict(torch.load("./models/neural_network_2021_05_30.pth"))

# Move tensors to GPU if CUDA is available
if train_on_gpu:
    neural_network_model.cuda()

# Switch the model to the prediction mode
neural_network_model.eval()


# Define transformation for image to be applicable for neural network
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# Get enrollment templates
templates = load_templates('./enrollment')
biometric_distances = []
user_name = "unknown"

# Use index = 0 for built-in webcam and index = 1 for external webcam,
# i.e. Logitech C922
idx = 1
cap = cv2.VideoCapture(index=idx)
detector = dlib.get_frontal_face_detector()

eyes_states = []
eyes_states_chain = []

neural_network_class = "unknown"
neural_network_predictions = []
neural_network_prediction = "unknown"

texture_class = "unknown"
texture_predictions = []
texture_prediction = "unknown"

grey_threshold = 0.92

# Choose filter width as an odd number.
# The greater number you choose, the more slow-response program behavior will be
filter_width = 5

# If eyes states chain length is 5, it should be finished with an "opened" state.
# If eyes states chain length is 4, it should also be finished with an "opened" state.
eyes_states_chain_length = 5

frame_counter = 0
frame_with_face_counter = 0

check_passed = False
machine_learning_prediction = "unknown"

nn_qty_unknown = 0
nn_qty_face = 0
nn_qty_photo = 0

# Measured frame rate is 12 fps. Minimal blink rate is about 10 blinking per minute.
# So, one blinking duration is about 6 seconds or 72 frames. With some margin
# we take frame quantity about 90 frames.
max_frame_counter = 90

while True:
    success, frame = cap.read()

    frame_height = np.size(frame, 0)
    frame_width = np.size(frame, 1)

    faces = detector(frame)

    if len(faces) == 0:
        msg = "No face is found."
    elif len(faces) == 1:
        msg = "One face is found. It's OK."
    else:
        msg = "{} faces are found. It's too much".format(len(faces))
    cv2.putText(frame, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if len(faces) == 1:
        # There should be one and only one face on the frame
        # In this case we increase number of frames with face
        frame_with_face_counter += 1

        face = faces[0]

        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # Now try to use neural network classifier
        if (0 < x1) and (x1 < frame_width) and (0 < x2) and (x2 < frame_width) and (x1 < x2) and \
                (0 < y1) and (y1 < frame_height) and (0 < y2) and (y2 < frame_height) and (y1 < y2):

            # Crop the frame before sending it to the neural network
            cropped_frame = frame[y1:y2, x1:x2]

            # Convert the frame from BGR to RGB
            cropped_frame_in_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

            # Convert the frame to PIL (Python Imaging Library) format
            pil_cropped_frame_in_rgb = Image.fromarray(cropped_frame_in_rgb)

            input_from_frame = transform(pil_cropped_frame_in_rgb)

            # # Convert the frame from BGR to grayscale
            # cropped_frame_in_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
            #
            # # Convert the frame to PIL (Python Imaging Library) format
            # pil_cropped_frame_in_gray = Image.fromarray(cropped_frame_in_gray)

            # # Transform image resolution to values which are applicable for neural network
            # input_from_frame = transform(pil_cropped_frame_in_gray)

            # Add the forth dimension to tensor
            input_from_frame = input_from_frame[None, ...]

            # Transform data to cuda format
            input_from_frame = input_from_frame.cuda()

            # Switch the model to the prediction mode
            with torch.no_grad():
                output_from_frame = neural_network_model(input_from_frame)

            # Transform tensor to python list
            output_from_frame = output_from_frame.tolist()[0]

            # Get the image class as an index of list element with maximum value
            output_class = np.argmax(output_from_frame)

            neural_network_class = "unknown"
            if output_class == 1:
                neural_network_class = "face"
            elif output_class == 0:
                neural_network_class = "photo"

            # Try to detect grayscale image against RGB image

            # Split r, g, b channels
            r, g, b = cv2.split(cropped_frame_in_rgb)

            # Get differences between (b,g), (r,g), (b,r) channel pixels
            r_g = np.count_nonzero(abs(r - g))
            r_b = np.count_nonzero(abs(r - b))
            g_b = np.count_nonzero(abs(g - b))

            # Get sum of differences
            diff_sum = float(r_g + r_b + g_b)

            # Find ratio of diff_sum with respect to size of image
            ratio = diff_sum / cropped_frame_in_rgb.size

            if ratio > grey_threshold:
                texture_class = "color"
            else:
                texture_class = "grey"
            print("ratio is {0}".format(ratio))

        # # Use random forest classifier on texture features
        # face_locations = face_recognition.face_locations(frame)
        # if len(face_locations) == 1:
        #     template = face_recognition.face_encodings(frame)[0]
        #     template = template.reshape(1, -1)
        #     texture_class = texture_random_forest_model.predict(template)[0]

        landmarks = predictor(frame, face)

        # for n in range(0, 68):
        #     x = landmarks.part(n).x
        #     y = landmarks.part(n).y
        #     cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

        # Right eye processing
        right_eye = []
        all_x = 0
        all_y = 0
        for n in range(36, 42):
            right_eye.append(landmarks.part(n))
            all_x += landmarks.part(n).x
            all_y += landmarks.part(n).y
            cv2.circle(frame, (landmarks.part(n).x, landmarks.part(n).y), 3, (255, 0, 0), -1)
        center_x = int(all_x / 6)
        center_y = int(all_y / 6)
        right_eye_center = {'x': center_x, 'y': center_y}

        right_eye_width = get_distance(landmarks.part(36), landmarks.part(39))

        right_eye_out_angle = get_angle(landmarks.part(37), landmarks.part(36), landmarks.part(41))
        right_eye_in_angle = get_angle(landmarks.part(38), landmarks.part(39), landmarks.part(40))

        right_eye_in_dist = get_distance(landmarks.part(38), landmarks.part(40))
        right_eye_normed_in_dist = round(right_eye_in_dist / right_eye_width, 3)

        right_eye_out_dist = get_distance(landmarks.part(37), landmarks.part(41))
        right_eye_normed_out_dist = round(right_eye_out_dist / right_eye_width, 3)

        right_eye_square = get_eye_square(right_eye, right_eye_center)
        right_eye_normed_square = round(right_eye_square / (right_eye_width ** 2), 4)

        right_eye_perimeter = get_perimeter(right_eye)
        right_eye_normed_perimeter = round(right_eye_perimeter / right_eye_width, 3)

        cv2.circle(frame, (right_eye_center['x'], right_eye_center['y']), 3, (0, 0, 255), -1)

        # Left eye processing
        left_eye = []
        all_x = 0
        all_y = 0
        for n in range(42, 48):
            left_eye.append(landmarks.part(n))
            all_x += landmarks.part(n).x
            all_y += landmarks.part(n).y
            cv2.circle(frame, (landmarks.part(n).x, landmarks.part(n).y), 3, (255, 0, 0), -1)
        center_x = int(all_x / 6)
        center_y = int(all_y / 6)
        left_eye_center = {'x': center_x, 'y': center_y}

        left_eye_width = get_distance(landmarks.part(42), landmarks.part(45))

        left_eye_out_angle = get_angle(landmarks.part(44), landmarks.part(45), landmarks.part(46))
        left_eye_in_angle = get_angle(landmarks.part(43), landmarks.part(42), landmarks.part(47))

        left_eye_in_dist = get_distance(landmarks.part(43), landmarks.part(47))
        left_eye_normed_in_dist = round(left_eye_in_dist / left_eye_width, 3)

        left_eye_out_dist = get_distance(landmarks.part(44), landmarks.part(46))
        left_eye_normed_out_dist = round(left_eye_out_dist/left_eye_width, 3)

        left_eye_square = get_eye_square(left_eye, left_eye_center)
        left_eye_normed_square = round(left_eye_square / (left_eye_width ** 2), 4)

        left_eye_perimeter = get_perimeter(left_eye)
        left_eye_normed_perimeter = round(left_eye_perimeter / left_eye_width, 3)

        cv2.circle(frame, (left_eye_center['x'], left_eye_center['y']), 3, (0, 0, 255), -1)

        x_input = pd.DataFrame(list(zip([right_eye_normed_square],
                                        [right_eye_normed_perimeter],
                                        [right_eye_out_angle],
                                        [right_eye_in_angle],
                                        [right_eye_normed_out_dist],
                                        [right_eye_normed_in_dist],

                                        [left_eye_normed_square],
                                        [left_eye_normed_perimeter],
                                        [left_eye_out_angle],
                                        [left_eye_in_angle],
                                        [left_eye_normed_out_dist],
                                        [left_eye_normed_in_dist])),
                               columns=['r_square', 'r_perimeter', 'r_out_angle',
                                        'r_in_angle', 'r_out_dist', 'r_in_dist',
                                        'l_square', 'l_perimeter', 'l_out_angle',
                                        'l_in_angle', 'l_out_dist', 'l_in_dist'])

        eyes_state = random_forest_model.predict(x_input)[0]
        eyes_states.append(eyes_state)

        if len(eyes_states) > filter_width:
            eyes_states.pop(0)

        if len(eyes_states) == filter_width:
            sorted_eyes_states = eyes_states.copy()
            sorted_eyes_states.sort()
            eyes_state_number = int((filter_width - 1) / 2)
            filtered_eyes_state = sorted_eyes_states[eyes_state_number]
            cv2.putText(frame, "Eyes are {}.".format(filtered_eyes_state), (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

            if len(eyes_states_chain) == 0:
                eyes_states_chain.append(filtered_eyes_state)
            else:
                if eyes_states_chain[-1] != filtered_eyes_state:
                    eyes_states_chain.append(filtered_eyes_state)
                if ((len(eyes_states_chain) == eyes_states_chain_length)
                    and (eyes_states_chain[-1] == "opened")) or \
                        ((len(eyes_states_chain) == eyes_states_chain_length - 1)
                         and (eyes_states_chain[-1] == "opened")):
                    check_passed = True

    neural_network_predictions.append(neural_network_class)

    neural_network_counter = Counter(neural_network_predictions)

    nn_qty_unknown = neural_network_counter["unknown"]
    nn_qty_face = neural_network_counter["face"]
    nn_qty_photo = neural_network_counter["photo"]

    texture_predictions.append(texture_class)

    if frame_counter == max_frame_counter:
        if frame_with_face_counter < (max_frame_counter / 2):
            machine_learning_prediction = "unknown"
            neural_network_prediction = "unknown"
            texture_prediction = "unknown"
            user_name = "unknown"
        else:
            if check_passed:
                machine_learning_prediction = "face"
            else:
                machine_learning_prediction = "photo"

            neural_network_prediction = mode(neural_network_predictions)

            texture_prediction = mode(texture_predictions)

            # Code for biometric identification

            frame_in_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(frame_in_rgb)

            if len(face_locations) == 1:
                template = face_recognition.face_encodings(frame_in_rgb)[0]

                biometric_distances = face_recognition.face_distance(list(templates.values()), template)
                template_index = np.argmin(biometric_distances)

                identification_result = face_recognition.compare_faces(
                    [list(templates.values())[template_index]], template)[0]

                if machine_learning_prediction == "face" or machine_learning_prediction == "photo":
                    if identification_result:
                        user_name = list(templates.keys())[template_index]
                        user_name = user_name.capitalize()
                    else:
                        user_name = "unknown"
                elif machine_learning_prediction == "unknown":
                    user_name = "unknown"

        check_passed = False
        frame_counter = 0
        frame_with_face_counter = 0
        eyes_states_chain.clear()
        neural_network_predictions.clear()
        texture_predictions.clear()

    cv2.putText(frame, "ML prediction is {}.".format(machine_learning_prediction), (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    cv2.putText(frame, "NN prediction is {}.".format(neural_network_prediction), (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    cv2.putText(frame, "Color prediction is {}.".format(texture_prediction), (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    cv2.putText(frame, "User name is {}".format(user_name), (10, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    dt_now = datetime.datetime.now()
    current_time = dt_now.time()

    print("Frame {0:05d} - {1} - {2} - {3} - {4} - {5}".format(
        frame_counter, current_time, eyes_states_chain, check_passed,
        machine_learning_prediction, neural_network_prediction))

    print("Neural network predictions: ")
    print(neural_network_predictions)

    print("Neural network count: unknown - {0}, face - {1}, photo - {2}".format(
        nn_qty_unknown, nn_qty_face, nn_qty_photo))

    frame_counter += 1

    cv2.imshow("video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
