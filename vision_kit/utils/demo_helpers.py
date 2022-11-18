import os
from fileinput import filename
from threading import Thread

import cv2

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
VIDEO_EXT = [".mp4", ".avi"]


class ImageReader:
    """A wrapper that reads images from directory or image itself.
    """

    def __init__(self, image_path: str, save_path: str = None) -> None:
        self.image_path = image_path
        if os.path.isdir(self.image_path):
            self.files = self._get_image_list(self.image_path)
        else:
            self.files = [self.image_path]
        self.root_save_path = save_path
        if save_path:
            os.makedirs(save_path, exist_ok=True)

    def update(self) -> None:
        print("Image Path: \t", self.files[self.index], "\t", end="")
        self.image = cv2.imread(self.files[self.index])

    def stop(self) -> None:
        cv2.destroyAllWindows()

    def save(self, image: cv2.Mat, filename: str) -> None:
        filename = os.path.splitext(os.path.basename(filename))[0] + "_res.jpg"
        save_path = os.path.join(self.root_save_path, filename)
        cv2.imwrite(save_path, image)

    def __iter__(self):
        self.index = -1
        return self

    def __next__(self) -> cv2.Mat:
        self.index += 1
        if self.index > len(self.files) - 1:
            return False, None
        self.update()
        return True, self.image

    def __len__(self):
        return 0

    def _get_image_list(self, folder) -> list:
        image_paths: list = []
        for root, sub, file_list in os.walk(folder):
            for file in file_list:
                path = os.path.join(root, file)
                ext = os.path.splitext(path)[1]
                if ext in IMAGE_EXT:
                    image_paths.append(path)
        return image_paths


class VideoReader:
    """A wrapper that reads frames from cv2.VideoCapture 
    """

    def __init__(self, video_path: str = '0', width: int = 640, height: int = 480, use_thread: bool = True, save_path: str = None):
        self.pipe = eval(video_path) if video_path.isnumeric() else video_path
        self.cap = cv2.VideoCapture(self.pipe)

        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.org_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.org_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = None if video_path.isnumeric() else int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = width
        self.height = height

        self.use_thread = use_thread
        self.video_writer = None
        self.save_path = save_path

    def update(self):
        print("Frame ID: \t", self.count, "\t", end="")
        self.has_frame, self.frame = self.cap.read()

    def save(self, frame):
        if self.video_writer is None and self.save_path:
            self.video_writer = cv2.VideoWriter(
                self.save_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (frame.shape[1], frame.shape[0])
            )
        else:
            self.video_writer.write(frame)

    def stop(self):
        if self.video_writer:
            self.video_writer.release()
        self.cap.release()
        cv2.destroyAllWindows()

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1

        if self.use_thread:
            self.thread = Thread(target=self.update, args=(), daemon=True).start()
        else:
            self.update()

        return self.has_frame, self.frame

    def __len__(self):
        return 0
