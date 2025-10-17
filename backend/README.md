# BreatheSafe - AI-Powered Baby Breathing Monitor (Backend)

**Real-time object detection, breathing rate monitoring, and AI-powered sleep position classification using MediaPipe Pose and Computer Vision**

⚠️ **FOR RESEARCH/DEMO ONLY - NOT A MEDICAL DEVICE**

---

## 🏆 Hackathon Project

This is the **backend server** for BreatheSafe, an AI-powered baby breathing monitor that uses **real-time object detection** and computer vision to:
- 🎯 **Detect and track babies** in video using MediaPipe Pose (33 body landmarks)
- 📊 **Analyze breathing rate** (BPM) from chest/shoulder movement patterns
- 🛏️ **Classify sleep position** (safe/unsafe) using ML classifier
- 🤖 **Real-time pose estimation** with confidence scoring
- 📡 **Stream detection results** via WebSocket to mobile app

**Frontend**: React Native mobile app located at `/Desktop/breathe-safe-v1`

---

## 🚀 Quick Demo (For Judges)

Get the demo running in **3 simple steps**:

### Prerequisites
- Python 3.11 or 3.12 (Python 3.14+ not supported by MediaPipe)
- macOS, Windows, or Linux

### Step 1: Install Dependencies
```bash
# Navigate to backend folder
cd simple-mediapipe-project-main

# Create virtual environment with Python 3.12
python3.12 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows

# Install all dependencies
pip install opencv-python mediapipe "numpy>=1.24.0,<2.0.0" scipy websockets streamlit matplotlib scikit-learn joblib
```

### Step 2: Run the Video Analysis Server
```bash
# Analyze the demo video (baby_monitor.mp4) with looping
python3 -m breath_monitor.video_server \
  --video ../breathe-safe-v1/assets/videos/baby-monitor.mp4 \
  --loop

# Or use your own video file
python3 -m breath_monitor.video_server --video /path/to/your/video.mp4 --loop
```

**What you'll see:**
```
============================================================
Video Breathing Monitor Server - WebSocket Integration
============================================================
[WARNING] NOT A MEDICAL DEVICE - FOR RESEARCH/DEMO ONLY
[INFO] Analyzing video: ../breathe-safe-v1/assets/videos/baby-monitor.mp4

✅ Position classifier loaded
[OK] Video loaded: 1920x1080 @ 30.0fps
[OK] Duration: 177.0s (5304 frames)
[OK] Loop mode: Enabled (video will repeat)
[INFO] WebSocket server started on ws://0.0.0.0:8765
[INFO] Server ready! Connect your React Native app to ws://localhost:8765

[000246] BPM:   22.1 | Position:     SAFE (safe) | Confidence: 0.98 | Clients: 0
[000283] BPM:   16.7 | Position:  UNKNOWN (danger) | Confidence: 0.98 | Clients: 0
[001214] BPM:   18.3 | Position:     SAFE (safe) | Confidence: 1.00 | Clients: 0
```

### Step 3: Connect the Frontend (Optional)
See the **Frontend README** at `/Desktop/breathe-safe-v1/README.md` for instructions to run the React Native mobile app.

The app will automatically connect to `ws://localhost:8765` and display:
- Real-time breathing rate
- Position safety indicators
- Historical data charts
- AI-powered recommendations

---

## 📋 Features

### 🎥 Object Detection & Computer Vision Analysis
- **Real-time Object Detection**: MediaPipe Pose detects and tracks person in frame
- **33-Point Landmark Detection**: Full body pose estimation with confidence scores
- **Breathing Rate Analysis**: Tracks chest/shoulder landmark movement patterns
- **32-Point Chest Monitoring**: Multi-point optical flow tracking for breathing
- **Signal Processing**: Bandpass filtering, peak detection, and noise reduction
- **Real-time Performance**: Processes 30 FPS video streams with <50ms latency

### 🤖 Machine Learning
- **Position Classifier**: Trained model detects safe (back) vs unsafe (stomach) positions
- **Training Dataset**: 178 labeled images (109 safe, 69 unsafe positions)
- **High Accuracy**: 90%+ position classification accuracy
- **Model File**: `models/position_model.joblib` (scikit-learn Random Forest)

### 📡 Integration
- **WebSocket Server**: Broadcasts data on port 8765
- **JSON Messages**: Real-time breathing rate, position, confidence scores
- **Auto-Reconnect**: Resilient connection handling
- **Multi-Client**: Supports multiple concurrent connections

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Video Input Source                        │
│  (baby_monitor.mp4 or live camera feed)                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              MediaPipe Pose Detection                        │
│  • 33 body landmarks tracked                                 │
│  • Chest ROI identification                                  │
│  • 32-point feature tracking                                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌────────────────────┐  ┌─────────────────────┐
│  Breathing Analyzer│  │ Position Classifier │
│  • Signal filtering│  │ • ML Model (joblib) │
│  • Peak detection  │  │ • Safe/Unsafe       │
│  • BPM calculation │  │ • Confidence score  │
└─────────┬──────────┘  └──────────┬──────────┘
          │                        │
          └───────────┬────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              WebSocket Broadcaster                           │
│  ws://0.0.0.0:8765                                          │
│  • JSON messages @ 1Hz                                       │
│  • breathing_rate, position_safe, confidence                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              React Native Mobile App                         │
│  • Real-time dashboard                                       │
│  • Historical charts                                         │
│  • AI recommendations                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
simple-mediapipe-project-main/
├── breath_monitor/              # Main package
│   ├── video_server.py          # ⭐ Video analysis server (DEMO THIS!)
│   ├── server.py                # Live camera server
│   ├── pose_backend.py          # MediaPipe integration + ML classifier
│   ├── signal.py                # Breathing analysis algorithms
│   ├── events.py                # WebSocket broadcaster
│   ├── video_capture.py         # Video file processing
│   ├── capture.py               # Camera capture
│   ├── draw.py                  # Visualization utilities
│   ├── cli.py                   # CLI interface
│   ├── ui_streamlit.py          # Web UI (alternative interface)
│   └── train_pose_classifier.py # ML model training script
├── models/
│   └── position_model.joblib    # Trained position classifier
├── docs/
│   ├── BREATHING_README.md      # Technical deep dive
│   ├── algorithm.md             # Algorithm documentation
│   └── validation.md            # Validation protocol
├── README.md                    # This file
├── README_POSITION_CLASSIFIER.md # ML model documentation
├── CHANGELOG.md                 # Version history
├── CONTRIBUTING.md              # Contribution guidelines
├── requirements.txt             # Python dependencies
└── pyproject.toml               # Package configuration
```

---

## 🔬 How It Works

### 1. Object Detection & Pose Estimation
Uses **MediaPipe Pose Landmarker** for real-time object detection:
- Detects person/baby in video frame
- Identifies 33 body landmarks (shoulders, hips, chest, etc.)
- Outputs normalized landmark coordinates (x, y) with confidence scores
- Runs at 30 FPS on standard hardware

### 2. Breathing Rate Analysis
- Tracks shoulder/chest vertical movement (y-coordinate changes)
- Applies bandpass filter (0.08-1.2 Hz = 5-72 BPM)
- Detects peaks with hysteresis (minimum 2s inter-breath interval)
- Calculates BPM from median inter-peak intervals
- Smooths for display with exponential moving average

### 3. Position Classification (ML-based)
- Extracts detected pose landmarks (shoulders, hips, chest, spine)
- Normalizes landmark coordinates for scale invariance
- Feeds 66 features (33 landmarks × 2D) to trained Random Forest classifier
- Classifies pose as: `safe` (back sleeping) or `danger` (stomach sleeping)
- Outputs prediction with confidence score (0.0-1.0)
- Model trained on 178 labeled images (90%+ accuracy)

### 4. Data Broadcasting
- Packages breathing rate + position data into JSON
- Broadcasts via WebSocket at 1 Hz
- Format:
```json
{
  "breathing_rate": 18.5,
  "confidence": 0.98,
  "pose_detected": true,
  "position_safe": true,
  "position_label": "safe",
  "position_confidence": 0.99,
  "classifier_available": true,
  "timestamp": 1697520000.123
}
```

---

## 🎯 Key Algorithms

### Breathing Detection
- **Input**: Video frames (30 FPS)
- **Signal**: Shoulder Y-position time series
- **Filtering**: 4th-order Butterworth bandpass (0.08-1.2 Hz)
- **Peak Detection**: Hysteresis with 2-second minimum inter-breath interval
- **Output**: BPM (breaths per minute)

### Position Classification
- **Training Data**: 178 images (109 safe, 69 dangerous positions)
- **Features**: Normalized pose landmark coordinates (33 points × 2D)
- **Model**: Random Forest Classifier (scikit-learn)
- **Accuracy**: 90%+ on test set
- **Inference**: Real-time classification per frame

See **[docs/algorithm.md](docs/algorithm.md)** for mathematical details.

---

## 📊 WebSocket API

Connect to `ws://localhost:8765` to receive real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8765');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('BPM:', data.breathing_rate);
  console.log('Position:', data.position_safe ? 'SAFE' : 'UNSAFE');
  console.log('Confidence:', data.confidence);
};
```

**Message Format**:
| Field | Type | Description |
|-------|------|-------------|
| `breathing_rate` | float | BPM (breaths per minute) |
| `confidence` | float | Detection confidence (0.0-1.0) |
| `pose_detected` | bool | Whether person detected in frame |
| `position_safe` | bool/null | true=safe, false=unsafe, null=unknown |
| `position_label` | string | "safe", "danger", or "unknown" |
| `position_confidence` | float | Position classification confidence |
| `classifier_available` | bool | Whether ML model loaded |
| `timestamp` | float | Unix timestamp |

---

## 🛠️ Advanced Usage

### Use Live Camera Instead of Video
```bash
# Analyze live webcam feed
python3 -m breath_monitor.server --camera 0
```

### Train Your Own Position Classifier
```bash
# Add training images to breath_monitor/data/safe/ and breath_monitor/data/danger/
# Then train:
python3 -m breath_monitor.train_pose_classifier

# Model will be saved to models/position_model.joblib
```

### Streamlit Web UI
```bash
# Launch interactive web interface
streamlit run breath_monitor/ui_streamlit.py
```

---

## 📚 Documentation

- **[docs/BREATHING_README.md](docs/BREATHING_README.md)** - Comprehensive technical documentation
- **[docs/algorithm.md](docs/algorithm.md)** - Mathematical algorithms and signal processing
- **[docs/validation.md](docs/validation.md)** - Validation protocol and accuracy testing
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and updates
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Guidelines for contributors

---

## 🎬 Demo Video Analysis

The included `baby_monitor.mp4` video demonstrates:
- ✅ Safe back sleeping position detection (high confidence)
- ⚠️ Unsafe stomach position detection (high confidence)
- 📊 Breathing rate detection (10-22 BPM range)
- 🔄 Position transitions (safe ↔ unsafe)
- 📈 Real-time confidence scores

**Video Stats**:
- Resolution: 1920×1080 @ 30 FPS
- Duration: 177 seconds (5,304 frames)
- Content: Multiple position changes, breathing patterns

---

## 🔧 Troubleshooting

### WebSocket Event Loop Warning
You may see: `RuntimeError: no running event loop`

**This is normal and doesn't affect functionality.** The WebSocket server still works perfectly. It's a cosmetic logging issue with the asyncio initialization.

### Camera Permission Denied
If using live camera and seeing permission errors:
1. macOS: System Settings → Privacy & Security → Camera → Enable Python/Terminal
2. Windows: Settings → Privacy → Camera → Allow apps to access camera

### Low Detection Confidence
- Ensure good lighting
- Person should be clearly visible
- Camera positioned to see full torso
- Minimize occlusions (blankets covering landmarks)

---

## 🏁 System Requirements

### Minimum
- Python 3.11 or 3.12 (3.14+ not supported)
- 4GB RAM
- Webcam or video file
- macOS 10.14+, Windows 10+, or Linux

### Recommended
- Python 3.12
- 8GB RAM
- Good lighting conditions
- Modern multi-core CPU

---

## ⚠️ Known Limitations

This is a **research prototype** and **NOT a medical device**:
- ❌ Not FDA approved
- ❌ Not suitable for real medical monitoring
- ❌ Should not replace proper medical equipment
- ❌ For demonstration and research purposes only

**Technical Limitations**:
- Requires visible pose landmarks (chest/shoulders)
- Affected by lighting conditions
- Movement can contaminate signal
- 10-15 second calibration period on startup

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

- **MediaPipe** - Google's ML framework for pose detection
- **OpenCV** - Video capture and processing
- **scikit-learn** - Machine learning model training
- **WebSockets** - Real-time communication
- **NumPy & SciPy** - Signal processing and numerical operations

---

## 📧 Contact

**For hackathon judges**: Questions? Issues running the demo? Contact us!

**Project Type**: Hackathon Submission  
**Category**: AI/ML, Computer Vision, Healthcare Technology  
**Backend**: Python + MediaPipe + ML  
**Frontend**: React Native + WebSocket Integration

---

**Made with ❤️ for safer baby sleep monitoring**
