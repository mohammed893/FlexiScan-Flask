from base_estimator import BaseEstimator
from pose_detection import PostureDefinitions
from pose_detection import angle_calc
from pose_detection import VideoPreprocessing
from pose_detection import FRAME_SIZE
import mediapipe as mp
import cv2 as cv

class VideoEstimator(VideoPreprocessing):
    def __init__(self, frame_size=FRAME_SIZE ,static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5):
        super().__init__(frame_size=frame_size)
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils  # Utility for drawing

    def get_landmarks(self, video_path, show=False, draw=False):
        # Load and process the video
        video = self.load_video(video_path)
        processed_frames = []

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Resize the frame
            frame = self.resize_frame(frame)

            # Convert to RGB for MediaPipe processing
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                if draw:
                    # Draw landmarks on the frame
                    self.mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
                    )
                
                # Convert to BGR for displaying
                processed_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                
                if show:
                    cv.imshow("Pose Estimatorion", processed_frame)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break

                processed_frames.append((processed_frame, results.pose_landmarks))
            else:
                processed_frames.append((frame, None))

        video.release()
        cv.destroyAllWindows()
        return processed_frames