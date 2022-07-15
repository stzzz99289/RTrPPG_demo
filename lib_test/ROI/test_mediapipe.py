import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# cap = cv2.VideoCapture(0)
#
# with mp_face_mesh.FaceMesh(
#         max_num_faces=1,
#         refine_landmarks=True,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5) as face_mesh:
#     while cap.isOpened():
#         success, image = cap.read()
#         if not success:
#             print("Ignoring empty camera frame.")
#             # If loading a video, use 'break' instead of 'continue'.
#             continue
#
#         # To improve performance, optionally mark the image as not writeable to pass by reference.
#         image.flags.writeable = False
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(image)
#
#         # Draw the face mesh annotations on the image.
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 mp_drawing.draw_landmarks(
#                     image=image,
#                     landmark_list=face_landmarks,
#                     connections=mp_face_mesh.FACEMESH_TESSELATION,
#                     landmark_drawing_spec=None,
#                     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
#                 mp_drawing.draw_landmarks(
#                     image=image,
#                     landmark_list=face_landmarks,
#                     connections=mp_face_mesh.FACEMESH_CONTOURS,
#                     landmark_drawing_spec=None,
#                     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
#                 mp_drawing.draw_landmarks(
#                     image=image,
#                     landmark_list=face_landmarks,
#                     connections=mp_face_mesh.FACEMESH_IRISES,
#                     landmark_drawing_spec=None,
#                     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
#
#         # Flip the image horizontally for a selfie-view display.
#         cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
#         if cv2.waitKey(5) & 0xFF == 27:
#             break
#
# cap.release()

facemesh_typelst = ['FACEMESH_CONTOURS', 'FACEMESH_FACE_OVAL', 'FACEMESH_IRISES', 'FACEMESH_LEFT_EYE',
                    'FACEMESH_LEFT_EYEBROW', 'FACEMESH_LEFT_IRIS', 'FACEMESH_LIPS', 'FACEMESH_RIGHT_EYE',
                    'FACEMESH_RIGHT_EYEBROW', 'FACEMESH_RIGHT_IRIS', 'FACEMESH_TESSELATION']

cap = cv2.VideoCapture(0)

# image_name = 'sample.jpeg'
# image = cv2.imread(image_name)
# height, width = image.shape[:2]

# custom_style_connection = mp_drawing_styles.get_default_face_mesh_tesselation_style()
# custom_style_connection.thickness = 3
# custom_style_connection.color = (0, 0, 256)
#
# custom_style_landmark = mp_drawing_styles.get_default_face_mesh_tesselation_style()
# custom_style_landmark.thickness = 2
# custom_style_landmark.color = (256, 0, 0)

right_eye_idx = np.asarray([246, 161, 159, 158, 157, 173, 133, 153, 145, 144, 163, 7])
left_eye_idx = np.asarray([398, 384, 385, 386, 387, 388, 263, 390, 373, 374, 380, 381, 382])
lip_idx = np.asarray([61, 185, 40, 39, 37, 267, 269, 270, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146])
face_oval_idx = np.asarray(
    [67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
     176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103])
eyes_lip_idx = [right_eye_idx, left_eye_idx, lip_idx]

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        height, width = image.shape[:2]

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        face_landmarks = results.multi_face_landmarks[0]
        landmarks_lst = face_landmarks.landmark

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        roi_corners_face = []
        for corner_index in face_oval_idx:
            landmark = landmarks_lst[corner_index]
            corner_point = np.asarray([landmark.x * width, landmark.y * height])
            roi_corners_face.append(corner_point)

        mask = np.zeros(image.shape, dtype=np.uint8)
        roi_corners = np.array([roi_corners_face], dtype=np.int32)
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
        cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)
        masked_image = cv2.bitwise_and(image, mask)

        roi_corners_leye = []
        roi_corners_reye = []
        roi_corners_lip = []
        for corner_index in left_eye_idx:
            landmark = landmarks_lst[corner_index]
            corner_point = np.asarray([landmark.x * width, landmark.y * height])
            roi_corners_leye.append(corner_point)
        for corner_index in right_eye_idx:
            landmark = landmarks_lst[corner_index]
            corner_point = np.asarray([landmark.x * width, landmark.y * height])
            roi_corners_reye.append(corner_point)
        for corner_index in lip_idx:
            landmark = landmarks_lst[corner_index]
            corner_point = np.asarray([landmark.x * width, landmark.y * height])
            roi_corners_lip.append(corner_point)

        roi_corners_leye = np.array([roi_corners_leye], dtype=np.int32)
        roi_corners_reye = np.array([roi_corners_reye], dtype=np.int32)
        roi_corners_lip = np.array([roi_corners_lip], dtype=np.int32)

        ignore_mask_color = (0,) * channel_count

        mask = np.zeros(image.shape, dtype=np.uint8)
        mask.fill(255)
        cv2.fillConvexPoly(mask, roi_corners_leye, ignore_mask_color)
        masked_image = cv2.bitwise_and(masked_image, mask)

        mask = np.zeros(image.shape, dtype=np.uint8)
        mask.fill(255)
        cv2.fillConvexPoly(mask, roi_corners_reye, ignore_mask_color)
        masked_image = cv2.bitwise_and(masked_image, mask)

        mask = np.zeros(image.shape, dtype=np.uint8)
        mask.fill(255)
        cv2.fillConvexPoly(mask, roi_corners_lip, ignore_mask_color)
        masked_image = cv2.bitwise_and(masked_image, mask)

        cv2.imshow("ROI visualization", masked_image)
        cv2.imshow("Original Image", image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break


    #
    # image.flags.writeable = True
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # mp_drawing.draw_landmarks(
    #     image=image,
    #     landmark_list=face_landmarks,
    #     connections=mp_face_mesh.FACEMESH_FACE_OVAL,
    #     landmark_drawing_spec=custom_style_landmark,
    #     connection_drawing_spec=custom_style_connection)
    #
    # mp_drawing.draw_landmarks(
    #     image=image,
    #     landmark_list=face_landmarks,
    #     connections=mp_face_mesh.FACEMESH_LEFT_EYE,
    #     landmark_drawing_spec=None,
    #     connection_drawing_spec=custom_style_connection)
    #
    # mp_drawing.draw_landmarks(
    #     image=image,
    #     landmark_list=face_landmarks,
    #     connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
    #     landmark_drawing_spec=None,
    #     connection_drawing_spec=custom_style_connection)
    #
    # mp_drawing.draw_landmarks(
    #     image=image,
    #     landmark_list=face_landmarks,
    #     connections=mp_face_mesh.FACEMESH_LIPS,
    #     landmark_drawing_spec=None,
    #     connection_drawing_spec=custom_style_connection)
    #
    # file_name = 'res/sample_' + 'ROI' + '.jpeg'
    # print("writing " + file_name + '...')
    # cv2.imwrite(file_name, image)
