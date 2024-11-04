import cv2 as cv

FRAME_SIZE = (480, 640)

class imagePreprocessing:
    def __init__(self , frame_size = FRAME_SIZE, normalization=True):
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




class VideoPreprocessing:
    def __init__(self, frame_size=FRAME_SIZE, normalization=True):
        self.frame_size = frame_size
        self.normalization = normalization

    def load_video(self, video_path):
        """
        Load a video from a file path and return a video capture object.
        """
        video = cv.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Video not found or cannot be opened at {video_path}")
        return video

    def resize_frame(self, frame):
        resized_frame = cv.resize(frame, self.frame_size)
        return resized_frame

    def process(self, video_path):
        """
        Process the video frame by frame: load, resize, and convert to RGB.
        """
        video = self.load_video(video_path)
        processed_frames = []

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Resize the frame
            frame = self.resize_frame(frame)

            # Convert to RGB if needed
            if self.normalization:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            processed_frames.append(frame)

        video.release()
        return processed_frames

def main():
    processor = imagePreprocessing()
    img = processor.load_image(r"C:\Users\body0\OneDrive\Desktop\testPose.jpg")
    img = processor.resize_image(img)
    cv.imshow("image",img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()