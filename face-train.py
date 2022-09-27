#설치한 OpenCV 패키지 불러오기
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

vcp = cv2.VideoCapture(0, cv2.CAP_DSHOW)

model = cv2.face.LBPHFaceRecognizer_create()

training_data, labels = [], []

count = 0

while True:
    ret, my_image = vcp.read()

    if ret is True:

        gray = cv2.cvtColor(my_image, cv2.COLOR_BGR2GRAY)

        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(gray, 1.5, 5, 0, (20, 20))

        facesCnt = len(faces)

        if facesCnt == 1:
            count += 1

            x, y, w, h = faces[0]

            faces_image = gray[y:y + h, x:x + w]

            training_data.append(faces_image)
            labels.append(count)

            print(training_data)
            print(labels)

            model.train(training_data, np.array(labels))

            model.save("model/face-trainner.yml")

            cv2.rectangle(my_image, faces[0], (255, 0, 0), 4)

        else:
            print("얼굴 미검출")

        cv2.imshow("train_my_face", my_image)

    if cv2.waitKey(1) == 13 or count == 100:
        break

vcp.release()

cv2.destroyAllWindows()