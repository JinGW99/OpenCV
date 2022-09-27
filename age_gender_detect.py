import cv2

face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

age_net = cv2.dnn.readNetFromCaffe(
    "model/deploy_age.prototxt", "model/age_net.caffemodel")

gender_net = cv2.dnn.readNetFromCaffe(
    "model/deploy_gender.prototxt", "model/gender_net.caffemodel")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# 나이 예측결과에 대한 결과값 리스트
age_list = ["(0 ~ 2)", "(4 ~ 6)", "(8 ~ 12)", "(15 ~ 20)",
            "(25 ~ 32)", "(38 ~ 43)", "(48 ~ 53)", "(60 ~ 100)"]

gender_list = ['Male', 'Female']

image = cv2.imread("image/IU2.jpg", cv2.IMREAD_COLOR)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.equalizeHist(gray)


faces = face_cascade.detectMultiScale(gray, 1.1, 5, 0, (100, 100))

# 탐지된 얼굴 수만큼 반복 실행하기
for face in faces:

    # 얼굴영역 좌표
    x, y, w, h = face

    face_image = image[y:y + h, x:x + w]

    blob = cv2.dnn.blobFromImage(face_image, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    gender_net.setInput(blob)

    gender_preds = gender_net.forward()

    gender = gender_preds.argmax()

    age_net.setInput(blob)


    age_preds = age_net.forward()


    age = age_preds.argmax()

    cv2.rectangle(image, face, (255, 0, 0), 4)

    result = gender_list[gender] + " " + age_list[age]

    cv2.putText(image, result, (x, y - 15), 0, 1, (255, 0, 0), 2)

cv2.imshow("myFace", image)

cv2.waitKey(0)