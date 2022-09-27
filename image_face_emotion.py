import cv2

image = cv2.imread("image/iu.jpg", cv2.IMREAD_COLOR)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.equalizeHist(gray)

emoticon_image = cv2.imread("image/emoticon.png", cv2.IMREAD_COLOR)

if image is None: raise Exception("이미지 읽기 실패")

face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

# 얼굴 검출 수행(정확도 높이는 방법의 아래 파라미터를 조절함)
# 얼굴 검출은 히스토그램 평황화한 이미지 사용
# scaleFactor : 1.5
# minNeighbors : 인근 유사 픽셀 발견 비율이 5번 이상
# flags : 0 => 더이상 사용하지 않는 인자값
# 분석할 이미지의 최소 크기 : 가로 100, 세로 100
faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100,100))

facesCnt = len(faces)

print(len(faces))

if facesCnt > 0:

    for face in faces:
        x, y, w, h = face

        face_image = cv2.resize(emoticon_image, (w, h), interpolation=cv2.INTER_AREA)

        image[y:y + h, x:x + w] = face_image;

        cv2.imwrite("result/my_image_emoticon.jpg", image)

        cv2.imshow("emoticon_image", cv2.imread("result/my_image_emoticon.jpg", cv2.IMREAD_COLOR))

else:print("얼굴 미검출")



cv2.waitKey(0)