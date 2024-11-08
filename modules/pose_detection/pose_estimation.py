from preprocessing import imagePreprocessing , VideoPreprocessing
from body_angles import PostureDefinitions
import numpy as np
import mediapipe as mp
import cv2 as cv


#===============================================================================================================================================
class ImageEstimator(imagePreprocessing, PostureDefinitions) :
    def __init__(self, static_image_mode = True, model_complexity = 1, enable_segmentation=False, min_detection_confidence=0.5):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence
            )
        self.mp_drawing = mp.solutions.drawing_utils
        self.processor = imagePreprocessing()
        self.posture_definitions = PostureDefinitions()

    def get_landmarks(self, image , show = False, draw = False ):
        self.image = self.processor.process(image)
        self.results = self.pose.process(self.image)
        lmList = []
        for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = self.image.shape
                cx, cy = int(lm.x*w) , int(lm.y*h)
                lmList.append([id, cx, cy])
        
        if self.results.pose_landmarks:
            if draw:
                    self.mp_drawing.draw_landmarks(
                    self.image, self.results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
                    ) 
            self.image = cv.cvtColor(self.image, cv.COLOR_RGB2BGR)
            if show:
                cv.imshow("image",self.image)
                cv.waitKey(0)
            
        else:
            print("No pose landmarks detected.")
        
        return self.image, lmList 

    def get_angle(self, p1 , p2 , p3, draw = False):
        p1 = np.array(p1[:2])
        p2 = np.array(p2[:2])
        p3 = np.array(p3[:2])
        radians = np.arctan2(p3[1]-p2[1], p3[0]-p2[0]) - np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle >180.0:
            angle %= 360
        
        if draw:
            cv.putText(self.image, str(angle), 
                           (p2[1], p2[2]), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA
                                )
        return self.image, angle
    
    def get_all(self, image):
        
        self.image, landmarks = self.get_landmarks(image)
        angles = {}
        for posture_name, posture_definition in self.posture_definitions.postures.items():
            points = posture_definition.points
            try:
                # Access the three required points from landmarks
                p1, p2, p3 = landmarks[points[0]], landmarks[points[1]], landmarks[points[2]]
                _, angle = self.get_angle(p1[1:], p2[1:], p3[1:])
                angles[posture_name] = angle
            except IndexError:
                print(f"Warning: Could not calculate angle for {posture_name} due to missing landmarks.")
        return {
            "land marks": landmarks,
            "angles": angles
        }
        



#===============================================================================================================================================
class VideoEstimatoror(VideoPreprocessing):
    def __init__(self, frame_size=(480, 640) ,static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5):
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

def main():
    estimator = ImageEstimator()
    image, lms = estimator.get_landmarks(r"C:\Users\body0\OneDrive\Desktop\testPose.jpg",  draw=True)
    data = estimator.get_all(r"C:\Users\body0\OneDrive\Desktop\testPose.jpg")
    print("Landmarks:")
    for landmark in data["land marks"]:
        print(f"ID: {landmark[0]}, X: {landmark[1]}, Y: {landmark[2]}")

# Print angles for each posture
    print("\nPosture Angles:")
    for posture_name, angle in data["angles"].items():
        print(f"{posture_name}: {angle:.2f}°")
    cv.imshow("image", image)
    cv.waitKey(0)
    for lm in lms:
        print(lm)    

if __name__ == "__main__":
    main()