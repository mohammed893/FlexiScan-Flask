import cv2 as cv
from config import FRAME_SIZE

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