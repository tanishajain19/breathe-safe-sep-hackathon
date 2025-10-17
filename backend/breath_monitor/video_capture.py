"""
Video file capture for analyzing pre-recorded videos.
Supports looping for continuous monitoring.
"""

import cv2
import time
import os
from typing import Generator, Tuple
import numpy as np


class VideoCapture:
    """Video file capture with looping support."""
    
    def __init__(self, video_path: str, loop: bool = True, fps_override: int = None):
        """
        Initialize video file capture.
        
        Args:
            video_path: Path to video file
            loop: Whether to loop the video continuously
            fps_override: Override video FPS (None = use video's native FPS)
        """
        self.video_path = video_path
        self.loop = loop
        self.fps_override = fps_override
        self.cap = None
        self.running = False
        self.width = 0
        self.height = 0
        self.fps = 0
        
    def start(self) -> bool:
        """
        Start video capture.
        
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(self.video_path):
            print(f"[ERROR] Video file not found: {self.video_path}")
            return False
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"[ERROR] Could not open video file: {self.video_path}")
            return False
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        native_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = self.fps_override if self.fps_override else native_fps
        
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / native_fps if native_fps > 0 else 0
        
        self.running = True
        print(f"[OK] Video loaded: {self.video_path}")
        print(f"[OK] Resolution: {self.width}x{self.height} @ {native_fps:.1f}fps")
        print(f"[OK] Duration: {duration:.1f}s ({frame_count} frames)")
        if self.loop:
            print(f"[OK] Loop mode: Enabled (video will repeat)")
        
        return True
    
    def frames(self) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Generator yielding frames with monotonic timestamps.
        
        Yields:
            Tuple of (frame, timestamp)
        """
        frame_delay = 1.0 / self.fps if self.fps > 0 else 0.033  # Default 30fps
        
        while self.running:
            ret, frame = self.cap.read()
            
            # Handle end of video
            if not ret:
                if self.loop:
                    print("[INFO] Video ended, restarting from beginning...")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        print("[ERROR] Failed to restart video")
                        break
                else:
                    print("[INFO] Video ended")
                    break
            
            timestamp = time.monotonic()
            yield frame, timestamp
            
            # Control playback speed
            time.sleep(frame_delay)
    
    def stop(self):
        """Stop video capture and release resources."""
        self.running = False
        if self.cap:
            self.cap.release()
        print("[OK] Video capture stopped")


class CameraCapture:
    """Webcam capture with configuration and clean shutdown."""
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize camera capture.
        
        Args:
            camera_id: Camera device index
            width: Frame width
            height: Frame height
            fps: Target frames per second
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.running = False
        
    def start(self) -> bool:
        """
        Start camera capture.
        
        Returns:
            True if successful, False otherwise
        """
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"[ERROR] Could not open camera {self.camera_id}")
            return False
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        self.running = True
        print(f"[OK] Camera started: {self.width}x{self.height} @ {self.fps}fps")
        return True
    
    def frames(self) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Generator yielding frames with monotonic timestamps.
        
        Yields:
            Tuple of (frame, timestamp)
        """
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Failed to read frame")
                break
                
            timestamp = time.monotonic()
            yield frame, timestamp
    
    def stop(self):
        """Stop camera capture and release resources."""
        self.running = False
        if self.cap:
            self.cap.release()
        print("[OK] Camera stopped")


def main():
    """Test video capture with display."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test video capture")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--loop", action="store_true", help="Loop video continuously")
    args = parser.parse_args()
    
    capture = VideoCapture(args.video, loop=args.loop)
    
    if not capture.start():
        return
    
    print("Press 'q' to quit")
    
    try:
        for frame, timestamp in capture.frames():
            cv2.imshow("Video Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        capture.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

