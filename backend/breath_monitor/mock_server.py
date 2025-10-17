"""
Mock breathing monitor server for testing without camera.
Broadcasts simulated breathing and position data via WebSocket.
"""

import time
import random
import math
from breath_monitor.events import WebSocketBroadcaster


class MockBreathingServer:
    """Mock server that simulates breathing data."""
    
    def __init__(self):
        """Initialize the mock server."""
        print("[INFO] Initializing Mock Breathing Monitor Server...")
        self.broadcaster = WebSocketBroadcaster(host="0.0.0.0", port=8765)
        self.running = False
        self.frame_count = 0
        
    def start(self):
        """Start the server."""
        print("[INFO] Starting WebSocket server on ws://0.0.0.0:8765")
        self.broadcaster.start()
        time.sleep(1)
        
        print("\n" + "=" * 60)
        print("Mock Breathing Monitor Server - TEST MODE")
        print("=" * 60)
        print("\n[WARNING] NOT A MEDICAL DEVICE - FOR RESEARCH/DEMO ONLY")
        print("[INFO] Server ready! Connect your React Native app to ws://localhost:8765")
        print("[INFO] Broadcasting simulated data...")
        print("[INFO] Press Ctrl+C to stop")
        print("=" * 60)
        print()
        
        self.running = True
        return True
    
    def generate_realistic_data(self):
        """Generate realistic breathing and position data."""
        # Simulate breathing rate with some natural variation
        base_bpm = 35  # Normal infant breathing
        time_factor = time.time() / 10
        variation = math.sin(time_factor) * 3 + random.gauss(0, 1)
        bpm = max(25, min(50, base_bpm + variation))
        
        # Simulate confidence
        confidence = random.uniform(0.75, 0.95)
        
        # Simulate position (mostly safe, occasionally unsafe for testing)
        position_roll = random.random()
        if position_roll > 0.9:  # 10% chance of unsafe
            position_safe = False
            position_label = "danger"
            position_confidence = random.uniform(0.7, 0.9)
        else:
            position_safe = True
            position_label = "safe"
            position_confidence = random.uniform(0.8, 0.95)
        
        # Signal quality
        if confidence > 0.85:
            signal_quality = "high"
        elif confidence > 0.7:
            signal_quality = "medium"
        else:
            signal_quality = "low"
        
        return {
            "breathing_rate": round(bpm, 1),
            "confidence": round(confidence, 2),
            "signal_quality": signal_quality,
            "pose_detected": True,
            "position_safe": position_safe,
            "position_label": position_label,
            "position_confidence": round(position_confidence, 2),
            "classifier_available": True,
            "timestamp": time.time()
        }
    
    def run(self):
        """Main broadcasting loop."""
        if not self.start():
            return
        
        last_broadcast_time = time.time()
        
        try:
            while self.running:
                self.frame_count += 1
                current_time = time.time()
                
                # Broadcast at 1Hz
                if current_time - last_broadcast_time >= 1.0:
                    data = self.generate_realistic_data()
                    
                    # Use the broadcaster's method
                    self.broadcaster.broadcast_state(
                        bpm=data["breathing_rate"],
                        apnea=False,
                        shallow=False,
                        confidence=data["confidence"],
                        timestamp=data["timestamp"]
                    )
                    
                    # Log status
                    pos_str = "SAFE" if data["position_safe"] else "UNSAFE"
                    print(f"[{self.frame_count:06d}] BPM: {data['breathing_rate']:>6.1f} | "
                          f"Position: {pos_str:>8s} ({data['position_label']}) | "
                          f"Confidence: {data['confidence']:.2f} | "
                          f"Clients: {len(self.broadcaster.clients)}")
                    
                    last_broadcast_time = current_time
                
                time.sleep(0.1)  # Small sleep to prevent busy loop
                
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the server."""
        print("\n[INFO] Shutting down...")
        self.running = False
        self.broadcaster.stop()
        print("[OK] Shutdown complete")


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("Mock Breathing Monitor Server")
    print("=" * 60)
    print("\n[INFO] This is a TEST server that simulates breathing data")
    print("[INFO] Use this to test the React Native app integration")
    print("[INFO] For real monitoring, use the full server with camera\n")
    
    server = MockBreathingServer()
    server.run()


if __name__ == "__main__":
    main()

