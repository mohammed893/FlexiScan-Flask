import cv2 as cv
IMAGE_SIZE = (224, 224)

class imagePreprocessing:
    def __init__(self , image_size = IMAGE_SIZE, normalization=True):
        self.image_size = image_size
        
    def load_image(self, image_path ):
        """
        Load an image from a file path.
        """
        image = cv.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at {image_path}")
        
        return image
    
    def resize_image(self, image):
        resized_image = cv.resize(image, self.image_size)
        # rgb_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
        return resized_image
            

def main():
    processor = imagePreprocessing()
    img = processor.load_image(r"C:\Users\body0\OneDrive\Desktop\testPose.jpg")
    img = processor.resize_image(img)
    cv.imshow("image",img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()