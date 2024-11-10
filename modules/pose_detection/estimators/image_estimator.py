from modules.pose_detection.estimators.base_estimator import *
from modules.pose_detection.preprocessing.image_preprocessing import *
from modules.pose_detection.definitions.body_angles import *
import cv2 as cv

class ImageEstimator(imagePreprocessing, PostureDefinitions) :
    def __init__(self, static_image_mode = True, model_complexity = 1, enable_segmentation=False, min_detection_confidence=0.5):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence
            )
        self.mp_drawing = mp.solutions.drawing_utils
        self.imProcessor = imagePreprocessing()
        self.posture_definitions = PostureDefinitions()

    def get_landmarks(self, image , show = False, draw = False ):
        self.image = self.imProcessor.process(image) # load, resize, and convert to RGB.
        self.results = self.pose.process(self.image) # 
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
        angle = angle_calc(p1, p2, p3)
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
            "land_marks": landmarks,
            "angles": angles
        }


