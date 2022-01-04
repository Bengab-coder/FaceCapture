import os
import cv2 as cv
import numpy as np

folder_for_captured_faces = "Captured faces"


def create_folder(name):
    if not os.path.exists(name):
        os.mkdir(name)
        print(f"Successfully created folder: '{name}'")


create_folder(folder_for_captured_faces)
haarcascade = cv.CascadeClassifier("haarcascade_frontalface.xml")
n = 1
path = os.getcwd()
path += f"/{folder_for_captured_faces}"


def detect_faces(image):
    global n
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=6)

    blank_image = np.zeros(image.shape, dtype=np.uint8)

    for x, y, w, h in faces:
        cv.rectangle(blank_image, (x, y), (x + w, y + h), (255, 255, 255), -1)

    cv.imshow("Blank", blank_image)

    image = cv.bitwise_and(blank_image, image, mask=None)
    cv.imshow("Bitwise_and", image)

    if len(faces):
        cv.imwrite(f"{path}/face{n}.png", image)
        n += 1


camera = cv.VideoCapture(0)

while True:
    _, frame = camera.read()
    frame = cv.resize(frame, (700, 500))
    frame = cv.flip(frame, 1)

    detect_faces(frame)

    cv.imshow("Image", frame)

    if cv.waitKey(1) & 0xff == ord("d"):
        break
camera.release()
cv.destroyAllWindows()
