import numpy as np
import cv2 as cv
import mediapipe
from .estimators.base_estimator import BaseEstimator
from .estimators.image_estimator import ImageEstimator
from .estimators.video_estimator import VideoEstimator
from .preprocessing.image_preprocessing import imagePreprocessing
from .preprocessing.video_preprocessing import VideoPreprocessing
from .definitions.body_angles import PostureDefinitions
from .config import FRAME_SIZE

def angle_calc(self, p1 , p2 , p3, draw = False):
        p1 = np.array(p1[:2])
        p2 = np.array(p2[:2])
        p3 = np.array(p3[:2])
        radians = np.arctan2(p3[1]-p2[1], p3[0]-p2[0]) - np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle >180.0:
            angle %= 360

