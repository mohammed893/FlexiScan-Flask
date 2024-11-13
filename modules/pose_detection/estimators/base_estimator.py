import mediapipe as mp
import cv2 as cv

class BaseEstimator:
    def __init__(self, static_image_mode = True, model_complexity = 1, enable_segmentation=False, min_detection_confidence=0.5):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def get_landmarks_from_frame(self, frame, draw=False):
        """
        Detect landmarks in a single frame (image or video frame).
        
        Parameters:
        frame (numpy array): The input image or video frame.
        draw (bool): Option to draw landmarks.

        Returns:
        tuple: Processed frame and list of landmarks.
        """
        # Convert to RGB for MediaPipe processing
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        lmList = []

        if results.pose_landmarks:
            h, w, c = frame.shape
            for id, lm in enumerate(results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            if draw:
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
                )

        return frame, lmList