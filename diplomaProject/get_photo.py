import cv2

idx = 1
cap = cv2.VideoCapture(index=idx)
img_counter = 0
max_img_number = 2000

while True:
    success, frame = cap.read()
    assert isinstance(frame, object)

    # Choose "eyes_opened" to write face photos with opened eyes
    # or "eyes_closed" to write face photos with closed eyes
    img_name = "D:/ML/Stepik/NN_CV/russian_dataset/new_images/andrew_kitchen_new/img_{0:04d}.png".\
        format(img_counter)
    cv2.imwrite(img_name, frame)
    print("{0} written.".format(img_name))
    img_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q')\
            or img_counter >= max_img_number:
        cap.release()
        cv2.destroyAllWindows()
        break
