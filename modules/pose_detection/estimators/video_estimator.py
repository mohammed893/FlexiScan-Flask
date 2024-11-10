from modules.pose_detection.estimators.base_estimator import *
from modules.pose_detection.preprocessing.video_preprocessing import *
from modules.pose_detection.config import FRAME_SIZE
from modules.pose_detection.definitions.body_angles import *
import numpy as np
import cv2 as cv
class VideoEstimator(BaseEstimator):
    def __init__(self, frame_size=FRAME_SIZE, static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5):
        # Initialize base class
        super().__init__(static_image_mode=static_image_mode, model_complexity=model_complexity, enable_segmentation=enable_segmentation, min_detection_confidence=min_detection_confidence)
        self.frame_size = frame_size
        self.posture_definitions = PostureDefinitions()

    def get_landmarks(self, video_path, show=False, draw=False):
        # Load and process the video
        video = cv.VideoCapture(video_path)
        processed_frames = []
        all_landmarks = []  
        frame_count = 0

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            # Only process every 30th frame
            if frame_count % 30 != 0:
                frame_count += 1
                continue
            
            # Resize the frame to match the desired frame size
            frame = cv.resize(frame, self.frame_size)
            
            # Get landmarks from the frame using the BaseEstimator method
            processed_frame, landmarks = self.get_landmarks_from_frame(frame, draw)
            if landmarks:
                all_landmarks.append(landmarks)
            
            processed_frames.append(processed_frame)
            
            if show:
                cv.imshow("Pose Estimation", processed_frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            # If we have enough frames to calculate the median
            if len(all_landmarks) == 30:  # Adjust batch size for calculating median
                break# Calculate the median of the landmarks
        
        video.release()
        cv.destroyAllWindows()
        
        median_landmarks = self.calculate_median_landmarks(all_landmarks)
        
        return processed_frames, median_landmarks


    def calculate_median_landmarks(self, all_landmarks):
        """
        Calculate the median of each landmark across all frames.
        
        Parameters:
        all_landmarks (list of list): A list of landmarks from each frame, where each entry is a list of landmarks.
        
        Returns:
        list: Median of landmarks (x, y) coordinates.
        """
        median_landmarks = []
        if all_landmarks:
            # Assuming all frames have the same number of landmarks
            num_landmarks = len(all_landmarks[0])  # Number of landmarks in each frame
            
            for idx in range(num_landmarks):
                # Collect x, y coordinates for the current landmark across all frames
                x_coords = [frame_landmarks[idx][1] for frame_landmarks in all_landmarks if len(frame_landmarks) > idx]
                y_coords = [frame_landmarks[idx][2] for frame_landmarks in all_landmarks if len(frame_landmarks) > idx]
                
                # Calculate median for x and y coordinates
                median_x = np.median(x_coords)
                median_y = np.median(y_coords)
                
                # Add the median (landmark index, median x, median y)
                median_landmarks.append([idx, median_x, median_y])

        return median_landmarks

    def get_all(self, video_path, show=False, draw=False):
        _, median_landmarks = self.get_landmarks(video_path, show=show, draw=draw)
        
        print(median_landmarks)
        angles = {}
        for posture_name, posture_definition in self.posture_definitions.postures.items():
            points = posture_definition.points
            try:
                # Access the three required points from landmarks
                p1, p2, p3 = median_landmarks[points[0]], median_landmarks[points[1]], median_landmarks[points[2]]
                angle = angle_calc(p1[1:], p2[1:], p3[1:])
                angles[posture_name] = angle
            except IndexError:
                print(f"Warning: Could not calculate angle for {posture_name} due to missing landmarks.")
        return {
            "land marks": median_landmarks,
            "angles": angles
        }

