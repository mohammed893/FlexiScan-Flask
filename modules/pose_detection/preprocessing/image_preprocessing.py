import cv2 as cv
from config import FRAME_SIZE

frameSize = FRAME_SIZE
class imagePreprocessing:
    def __init__(self , frame_size = frameSize, normalization=True):
        self.frame_size = frame_size
        
    def load_image(self, image_path ):
        """
        Load an image from a file path.
        """
        image = cv.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at {image_path}")
        
        return image
    
    def resize_image(self, image):
        resized_image = cv.resize(image, self.frame_size)
        # rgb_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
        return resized_image
    
    def process(self, image):
        """
        Process the image: load, resize, and convert to RGB.
        """
        image = self.load_image(image)
        image = self.resize_image(image)
        RGBImage = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        return RGBImage