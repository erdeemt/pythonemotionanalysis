import cv2
#RGB TEST
image = cv2.imread('saat.jpg')
"""
image = cv2.imread('rgb.jpeg')
B, G, R = cv2.split(image)
cv2.imshow("original", image)
cv2.waitKey(0)

cv2.imshow("blue", B)
cv2.waitKey(0)

cv2.imshow("Green", G)
cv2.waitKey(0)

cv2.imshow("red", R)
cv2.waitKey(0)

cv2.destroyAllWindows()
"""
#ARITMETHICS ON IMAGES
"""       
image1 = cv2.imread('inp1.jpg')
image2 = cv2.imread('inp2.jpg')

weightedSum = cv2.addWeighted(image1, 0.5, image2, 0.4, 0)

cv2.imshow('Weighted Image', weightedSum)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""
#RESIZING
"""
bigger = cv2.resize(image, (1050, 1610))

stretch_near = cv2.resize(image, (780, 540),
                          interpolation=cv2.INTER_LINEAR)

cv2.imshow("ORIGINAL",image )
cv2.waitKey(0)
cv2.imshow("BIGGER", bigger)
cv2.waitKey(0)
cv2.imshow("STRETCHNEAR", stretch_near)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
##GAUSSIAN BLURING
"""
Gaussian = cv2.GaussianBlur(image, (7, 7), 0)
cv2.imshow('Gaussian Blurring', Gaussian)
cv2.waitKey(0) 
"""
##ROTATING
"""
image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow("DONMUS", image)
cv2.waitKey(0)
image = cv2.rotate(image, cv2.ROTATE_180)
cv2.imshow("DONMUS", image)
cv2.waitKey(0)
image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imshow("DONMUS", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
##SMILE DETECTION


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_smile.xml')

faces  = face_cascade.detectMultiScale(image, 1.3, 5)


def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
    return frame


video_capture = cv2.VideoCapture(0)
while video_capture.isOpened():
    # Captures video_capture frame by frame
    _, frame = video_capture.read()

    # To capture image in monochrome
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calls the detect() function
    canvas = detect(gray, frame)

    # Displays the result on camera feed
    cv2.imshow('Video', canvas)

    # The control breaks once q key is pressed
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Release the capture once all the processing is done.
video_capture.release()
cv2.destroyAllWindows()


