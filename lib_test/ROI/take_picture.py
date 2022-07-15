import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    cv2.imshow("current frame", image)

    if cv2.waitKey(5) & 0xFF == ord('c'):
        image_path = 'sample_face.png'
        cv2.imwrite(image_path, image)
        break

cap.release()