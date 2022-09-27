import cv2

# 얼굴 탐지를 위한 모델
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

gender_net = cv2.dnn.readNetFromCaffe(
    "model/deploy_gender.prototxt", "model/gender_net.caffemodel")

# 학습된 모델데이터에 정의된 입력영상 각 채널에서 뺄 평균값
# 사용할 학습데이터는 반드시 아래와 같은 값을 사용해야 함
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# 성별 예측결과에 대한 결과값 리스트
gender_list = ["Male", "Female"]

# 분석하기 위한 이미지 불러오기
image = cv2.imread("image/my_face.jpg", cv2.IMREAD_COLOR)

# 흑백사진으로 변경
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 변환한 흑백사진으로부터 히스토그램 평활화
gray = cv2.equalizeHist(gray)

faces = face_cascade.detectMultiScale(gray, 1.1, 5, 0, (100, 100))

# 탐지된 얼굴 수만큼 반복 실행하기
for face in faces:
    # 얼굴영역 좌표
    x, y, w, h = face

    # 얼굴영역 이미지
    face_image = image[y:y + h, x:x + w]

    blob = cv2.dnn.blobFromImage(face_image, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    # 성별 예측하기
    # 분석 데이터를 입력하기
    gender_net.setInput(blob)

    # 성별 예측하기
    gender_preds = gender_net.forward()

    # 성별 예측 결과 가져오기(여러개 중에 1개 선택할 사용하는 알고리즘, TenserFlow의 Softmax와 유사)
    gender = gender_preds.argmax()

     # 얼굴 영역에 사각형 그리기
    cv2.rectangle(image, face, (255, 0, 0), 4)

    # 예측결과 문자열
    result = "Gender : "+ gender_list[gender]

    # 예측결과 문자열 사진에 추가하기
    cv2.putText(image, result, (x, y - 15), 0, 1, (255, 0, 0), 2)

# 이미지 출력
cv2.imshow("myFace", image)

# 입력받는 것 대기하기, 작성안하면, 결과창이 바로 닫힘
cv2.waitKey(0)