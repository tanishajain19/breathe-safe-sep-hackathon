"""
Integrated breathing monitor with WebSocket broadcasting.
Combines camera capture, pose detection, breathing analysis, position classification,
and WebSocket broadcasting for the BreatheSafe React Native app.
"""

import time
import argparse
from breath_monitor.capture import CameraCapture
from breath_monitor.pose_backend import PoseBackend
from breath_monitor.signal import BreathingAnalyzer
from breath_monitor.events import WebSocketBroadcaster


class BreathingMonitorServer:
    """Breathing monitor with WebSocket broadcasting."""
    
    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        """Initialize the server."""
        print("[INFO] Initializing Breathing Monitor Server...")
        
        # Initialize components
        self.capture = CameraCapture(camera_id, width, height, fps)
        self.pose = PoseBackend(num_trackers=32, tracker_refresh=0.5)
        self.analyzer = BreathingAnalyzer(
            window_sec=30.0,
            resp_low=0.08,
            resp_high=1.2,
            min_ibi_sec=2.0,
            apnea_sec=20.0
        )
        self.broadcaster = WebSocketBroadcaster(host="0.0.0.0", port=8765)
        
        self.running = False
        
    def start(self):
        """Start all components."""
        print("[INFO] Starting camera...")
        if not self.capture.start():
            print("[ERROR] Failed to start camera")
            return False
        
        print("[INFO] Starting WebSocket server on ws://0.0.0.0:8765")
        self.broadcaster.start()
        time.sleep(1)  # Wait for WebSocket server to initialize
        
        print("[INFO] Server ready! Connect your React Native app to ws://localhost:8765")
        print("[INFO] Press Ctrl+C to stop")
        print("=" * 60)
        
        self.running = True
        return True
    
    def process_and_broadcast(self):
        """Main processing loop."""
        frame_count = 0
        last_broadcast_time = time.time()
        
        try:
            for frame, timestamp in self.capture.frames():
                frame_count += 1
                
                # Pose detection with position classification
                pose_result = self.pose.infer(frame, timestamp)
                
                # Breathing analysis
                if pose_result["detected"] and pose_result["confidence"] > 0.3:
                    tracks_y = pose_result.get("tracks_y", [])
                    
                    # Use mid-shoulder position
                    value = 0.5
                    if pose_result.get("mid_shoulder_xy"):
                        _, y = pose_result["mid_shoulder_xy"]
                        value = y
                    
                    self.analyzer.add_sample(timestamp, value, tracks_y)
                
                # Analyze breathing
                analysis = self.analyzer.analyze(
                    visibility=pose_result.get("confidence", 0.0),
                    tracks_alive=pose_result.get("num_tracked_points", 0),
                    total_tracks=32
                )
                
                # Broadcast state at 1Hz
                current_time = time.time()
                if current_time - last_broadcast_time >= 1.0:
                    self.broadcast_state(pose_result, analysis, timestamp)
                    last_broadcast_time = current_time
                    
                    # Log status
                    bpm = analysis.get("bpm_smooth") or analysis.get("bpm")
                    bpm_str = f"{bpm:.1f}" if bpm else "--"
                    pos_label = pose_result.get("position_label", "unknown")
                    pos_safe = pose_result.get("position_safe")
                    pos_str = "SAFE" if pos_safe else "UNSAFE" if pos_safe is False else "UNKNOWN"
                    
                    print(f"[{frame_count:06d}] BPM: {bpm_str:>6s} | Position: {pos_str:>8s} ({pos_label}) | "
                          f"Confidence: {pose_result.get('confidence', 0):.2f} | "
                          f"Clients: {len(self.broadcaster.clients)}")
                
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        finally:
            self.stop()
    
    def broadcast_state(self, pose_result, analysis, timestamp):
        """Broadcast current state via WebSocket."""
        # Calculate breathing rate
        bpm = analysis.get("bpm_smooth") or analysis.get("bpm")
        
        # Format message for React Native app
        message = {
            "breathing_rate": float(bpm) if bpm else None,
            "confidence": float(pose_result.get("confidence", 0.0)),
            "timestamp": float(timestamp),
            "signal_quality": "high" if analysis.get("conf", 0) > 0.7 else "medium" if analysis.get("conf", 0) > 0.4 else "low",
            "pose_detected": bool(pose_result.get("detected", False)),
            "position_safe": pose_result.get("position_safe"),
            "position_label": pose_result.get("position_label", "unknown"),
            "position_confidence": float(pose_result.get("position_confidence", 0.0)),
            "classifier_available": bool(pose_result.get("classifier_available", False))
        }
        
        # Also include apnea/shallow for backwards compatibility with events.py
        self.broadcaster.broadcast_state(
            bpm=message["breathing_rate"],
            apnea=bool(analysis.get("apnea", False)),
            shallow=bool(analysis.get("shallow", False)),
            confidence=message["confidence"],
            timestamp=timestamp
        )
    
    def stop(self):
        """Stop all components."""
        print("\n[INFO] Shutting down...")
        self.running = False
        self.capture.stop()
        self.pose.close()
        self.broadcaster.stop()
        print("[OK] Shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Breathing Monitor Server with WebSocket Broadcasting"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Breathing Monitor Server - WebSocket Integration")
    print("=" * 60)
    print("\n[WARNING] NOT A MEDICAL DEVICE - FOR RESEARCH/DEMO ONLY\n")
    
    server = BreathingMonitorServer(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps
    )
    
    if server.start():
        server.process_and_broadcast()


if __name__ == "__main__":
    main()

