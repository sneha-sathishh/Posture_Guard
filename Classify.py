import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import math
import cv2
import numpy as np
from time import time
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'pose_landmarker_heavy.task'


base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)


def detectPose(image):
    landmarks_names = {"nose": 0, "left_eye_inner": 1, "left_eye": 2, "left_eye_outer": 3, "right_eye_inner": 4, "right_eye": 5, "right_eye_outer": 6, "left_ear": 7, "right_ear": 8, "mouth_left": 9, "mouth_right": 10, "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14, "left_wrist": 15, "right_wrist": 16, "left_pinky": 17, "right_pinky": 18, "left_index": 19, "right_index": 20, "left_thumb": 21, "right_thumb": 22, "left_hip": 23, "right_hip": 24, "left_knee": 25, "right_knee": 26, "left_ankle": 27, "right_ankle": 28, "left_heel": 29, "right_heel": 30, "left_foot_index": 31, "right_foot_index": 32}
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    try:
        pose_landmarks = detector.detect(mp_image).pose_landmarks[0]
        landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in pose_landmarks]
        left_shoulder_angle = calculateAngle(
            landmarks[landmarks_names["left_shoulder"]], landmarks[landmarks_names["left_elbow"]], landmarks[landmarks_names["left_wrist"]])
        right_shoulder_angle = calculateAngle(
            landmarks[landmarks_names["right_shoulder"]], landmarks[landmarks_names["right_elbow"]], landmarks[landmarks_names["right_wrist"]])
        left_elbow_angle = calculateAngle(
            landmarks[landmarks_names["left_elbow"]], landmarks[landmarks_names["left_wrist"]], landmarks[landmarks_names["left_shoulder"]])
        right_elbow_angle = calculateAngle(
            landmarks[landmarks_names["right_elbow"]], landmarks[landmarks_names["right_wrist"]], landmarks[landmarks_names["right_shoulder"]])
        neck_angle = calculateAngle(
            landmarks[landmarks_names["left_shoulder"]], landmarks[landmarks_names["right_shoulder"]], landmarks[landmarks_names["nose"]])
        
        angles = [left_shoulder_angle, right_shoulder_angle, left_elbow_angle, right_elbow_angle, neck_angle]
        return angles
    except Exception as e:
        print(e)
        return "No pose detected"
    

def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                         math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle


def inRange(angle, min, max):
    if angle < min or angle > max:
        return False
    return True





# def draw_landmarks_on_image(rgb_image, detection_result):
#     pose_landmarks_list = detection_result.pose_landmarks
#     annotated_image = np.copy(rgb_image)
#     for idx in range(len(pose_landmarks_list)):
#       pose_landmarks = pose_landmarks_list[idx] 
#       # Draw the pose landmarks.
#       pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#       pose_landmarks_proto.landmark.extend([
#         landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
#       ])
#       solutions.drawing_utils.draw_landmarks(
#         annotated_image,
#         pose_landmarks_proto,
#         solutions.pose.POSE_CONNECTIONS,
#         solutions.drawing_styles.get_default_pose_landmarks_style())
#     return annotated_image, pose_landmarks_list


if __name__ == '__main__':
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3, 1280)
    camera_video.set(4, 960)
    cv2.namedWindow('Sign Detection', cv2.WINDOW_NORMAL)
    while camera_video.isOpened():
        ok, frame = camera_video.read()
        time1 = time()
        if not ok:
            continue
        frame = cv2.flip(frame, 1) 
        pose = detectPose(frame)
        cv2.putText(frame, "Left Shoulder Angle: " + str(pose[0]), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Right Shoulder Angle: " + str(pose[1]), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Left Elbow Angle: " + str(pose[2]), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Right Elbow Angle: " + str(pose[3]), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Neck Angle: " + str(pose[4]), (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Sign Detection', frame)
        k = cv2.waitKey(1) & 0xFF
        if(k == 27):
            break
    camera_video.release()