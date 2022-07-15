import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# supported default ROI index set
facemesh_typelst = ['FACEMESH_CONTOURS', 'FACEMESH_FACE_OVAL', 'FACEMESH_IRISES', 'FACEMESH_LEFT_EYE',
                    'FACEMESH_LEFT_EYEBROW', 'FACEMESH_LEFT_IRIS', 'FACEMESH_LIPS', 'FACEMESH_RIGHT_EYE',
                    'FACEMESH_RIGHT_EYEBROW', 'FACEMESH_RIGHT_IRIS', 'FACEMESH_TESSELATION']
# print(mp_face_mesh.FACEMESH_FACE_OVAL)

# index of points of the ROI polyline
# note the order of the points
right_eye_idx = np.asarray([246, 161, 159, 158, 157, 173, 133, 153, 145, 144, 163, 7])
left_eye_idx = np.asarray([398, 384, 385, 386, 387, 388, 263, 390, 373, 374, 380, 381, 382])
lip_idx = np.asarray([61, 185, 40, 39, 37, 267, 269, 270, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146])
face_oval_idx = np.asarray(
    [67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
     176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103])


def get_ROI(image):
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        height, width = image.shape[:2]
        channel_count = image.shape[2]

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks_lst = face_landmarks.landmark  # landmark_lst[i] is a dict with information of landmark i

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # get (x,y) coordinates of roi corners
            roi_corners_face = []
            roi_corners_leye = []
            roi_corners_reye = []
            roi_corners_lip = []
            for corner_index in face_oval_idx:
                landmark = landmarks_lst[corner_index]
                corner_point = np.asarray([landmark.x * width, landmark.y * height])  # landmark coordinate is normalized
                roi_corners_face.append(corner_point)
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
            roi_corners_face = np.array([roi_corners_face], dtype=np.int32)
            roi_corners_leye = np.array([roi_corners_leye], dtype=np.int32)
            roi_corners_reye = np.array([roi_corners_reye], dtype=np.int32)
            roi_corners_lip = np.array([roi_corners_lip], dtype=np.int32)

            ROI_type = 'shadow'

            if ROI_type == 'holistic':
                # keep the face ROI and remove the rest
                # all 0 outside the polygon, all 255(0x11) inside the polygon
                ignore_mask_color = (255,) * channel_count
                mask = np.zeros(image.shape, dtype=np.uint8)
                cv2.fillConvexPoly(mask, roi_corners_face, ignore_mask_color)

                # remove the left eye, right eye and lip ROI and keep the rest
                # all 255 outside the polygon, all 0x00 inside the polygon
                ignore_mask_color = (0,) * channel_count
                cv2.fillConvexPoly(mask, roi_corners_leye, ignore_mask_color)
                cv2.fillConvexPoly(mask, roi_corners_reye, ignore_mask_color)
                cv2.fillConvexPoly(mask, roi_corners_lip, ignore_mask_color)

                # mask is all 255 in ROI, all 0 otherwise
                masked_image = cv2.bitwise_and(image, mask)
                return masked_image, mask
            elif ROI_type == 'shadow':
                ignore_mask_color = (255,) * channel_count
                mask = np.zeros(image.shape, dtype=np.uint8)
                cv2.fillConvexPoly(mask, roi_corners_face, ignore_mask_color)
                ignore_mask_color = (0,) * channel_count
                cv2.fillConvexPoly(mask, roi_corners_leye, ignore_mask_color)
                cv2.fillConvexPoly(mask, roi_corners_reye, ignore_mask_color)
                cv2.fillConvexPoly(mask, roi_corners_lip, ignore_mask_color)

                ignore_mask_color = (0,50,200)
                shadow_mask = np.zeros(image.shape, dtype=np.uint8)
                cv2.fillConvexPoly(shadow_mask, roi_corners_face, ignore_mask_color)
                ignore_mask_color = (0,) * channel_count
                cv2.fillConvexPoly(shadow_mask, roi_corners_leye, ignore_mask_color)
                cv2.fillConvexPoly(shadow_mask, roi_corners_reye, ignore_mask_color)
                cv2.fillConvexPoly(shadow_mask, roi_corners_lip, ignore_mask_color)

                masked_image_holistic = cv2.bitwise_and(image, shadow_mask)
                masked_image = cv2.addWeighted(image, 0.6, masked_image_holistic, 0.4, 0)
                return masked_image, mask

        else:
            mask = np.zeros(image.shape, dtype=np.uint8)
            mask.fill(255)
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), mask
