# BreatheSafe - AI-Powered Baby Breathing Monitor

**Real-time object detection and computer vision for baby sleep safety**

🏆 **Hackathon Submission** | 🎯 **Object Detection** | 🤖 **AI/ML** | 📱 **Mobile App**

---

## 🎥 Project Overview

BreatheSafe uses **real-time object detection** with MediaPipe Pose to monitor baby breathing and sleep positions:

- 🎯 **33-Point Pose Detection** - Tracks body landmarks in real-time @ 30 FPS
- 📊 **Breathing Rate Analysis** - Calculates BPM from chest/shoulder movement
- 🤖 **ML Position Classifier** - Detects safe (back) vs unsafe (stomach) positions
- 📱 **Mobile App** - Beautiful React Native interface with real-time monitoring
- 🧠 **AI Recommendations** - OpenAI GPT-4 powered sleep safety advice

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Backend Setup (Python Server)
```bash
cd backend

# Create Python 3.12 virtual environment
python3.12 -m venv venv
source venv/bin/activate  # macOS/Linux
# OR: venv\Scripts\activate  # Windows

# Install dependencies
pip install opencv-python mediapipe "numpy>=1.24.0,<2.0.0" scipy websockets streamlit matplotlib scikit-learn joblib

# Start video analysis server
python3 -m breath_monitor.video_server --video ../frontend/assets/videos/baby-monitor.mp4 --loop
```

**Expected Output:**
```
✅ Position classifier loaded
[OK] Video loaded: 1920x1080 @ 30.0fps
[INFO] WebSocket server started on ws://0.0.0.0:8765
[000246] BPM:   22.1 | Position:     SAFE (safe) | Confidence: 0.98
```

### Step 2: Frontend Setup (React Native App)
```bash
cd frontend

# Install dependencies
npm install

# Start Expo app
npx expo start

# Then press:
# - 'i' for iOS simulator
# - 'a' for Android emulator
# - Scan QR code for physical device
```

**The app will connect to the backend and display real-time monitoring!** ✅

---

## 🏗️ Project Structure

```
breathe-safe-sep-hackathon/
├── README.md                    # ⭐ This file
│
├── frontend/                    # React Native Mobile App
│   ├── README.md               # Frontend documentation
│   ├── App.js                  # Main navigation
│   ├── package.json            # Dependencies
│   ├── screens/                # Dashboard, History, Learn, Settings
│   ├── components/             # Reusable UI components
│   ├── services/               # WebSocket client
│   ├── context/                # State management
│   ├── utils/                  # Configuration
│   └── assets/
│       └── videos/
│           └── baby-monitor.mp4  # 🎬 Demo video (177s, 1920x1080)
│
└── backend/                     # Python MediaPipe Server
    ├── README.md                # Backend documentation
    ├── requirements.txt         # Python dependencies
    ├── breath_monitor/          # Main package
    │   ├── video_server.py      # ⭐ Video analysis server
    │   ├── pose_backend.py      # MediaPipe + ML classifier
    │   ├── signal.py            # Breathing analysis
    │   ├── events.py            # WebSocket broadcaster
    │   └── train_pose_classifier.py  # ML training
    ├── models/
    │   └── position_model.joblib  # Trained ML model
    ├── breath_monitor/data/       # Training dataset
    │   ├── safe/    (109 images)  # Safe positions
    │   └── danger/  (69 images)   # Unsafe positions
    └── docs/                      # Technical documentation
```

---

## 🎯 Key Features

### 🎥 Object Detection & Computer Vision
- **MediaPipe Pose**: Real-time detection of 33 body landmarks
- **30 FPS Processing**: Analyzes video frames in real-time
- **Confidence Scoring**: Each detection includes 0.0-1.0 confidence score
- **Multi-Point Tracking**: 32-point chest expansion monitoring

### 🤖 Machine Learning
- **Position Classifier**: Random Forest model (90%+ accuracy)
- **Training Dataset**: 178 labeled images (safe/unsafe positions)
- **Feature Engineering**: 66 normalized pose features (33 landmarks × 2D)
- **Real-time Inference**: <2ms classification time

### 📱 Mobile Application
- **Real-time Dashboard**: Live BPM and position indicators
- **Historical Charts**: Breathing patterns over time
- **AI Recommendations**: OpenAI GPT-4 personalized advice
- **Multi-language**: Educational content in 4 languages
- **Beautiful UX**: Calming pastel design for parents

---

## 🔬 Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│         Video Input (baby_monitor.mp4 or camera)            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              MediaPipe Pose Detection                        │
│  • Object detection (person in frame)                        │
│  • 33 body landmarks with confidence scores                  │
│  • Normalized coordinates (scale-invariant)                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴────────────┐
          ▼                        ▼
┌───────────────────┐    ┌──────────────────────┐
│ Breathing Analyzer│    │ Position Classifier  │
│ • Signal filtering│    │ • Random Forest ML   │
│ • Peak detection  │    │ • Safe/Unsafe        │
│ • BPM calculation │    │ • 90%+ accuracy      │
└─────────┬─────────┘    └──────────┬───────────┘
          │                         │
          └──────────┬──────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│          WebSocket Server (ws://localhost:8765)              │
│  • JSON messages @ 1Hz                                       │
│  • breathing_rate, position_safe, confidence scores          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            React Native Mobile App (Expo)                    │
│  • Real-time dashboard with breathing indicator              │
│  • Historical data charts and analytics                      │
│  • AI-powered recommendations (OpenAI GPT-4)                 │
│  • Educational content and safety alerts                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 What Makes This Special

### 1. **Real-Time Object Detection**
- MediaPipe Pose detects and tracks 33 body landmarks
- Processes 30 FPS video with <50ms latency
- Handles occlusion, rotation, and scale variations
- Confidence scoring for detection quality

### 2. **Computer Vision Breathing Analysis**
- Tracks shoulder/chest landmark Y-coordinates over time
- Advanced signal processing (bandpass filtering, peak detection)
- Calculates breathing rate from inter-peak intervals
- Hysteresis debouncing prevents false counts

### 3. **ML-Powered Safety Classification**
- Custom-trained Random Forest classifier
- 178-image training dataset (safe/unsafe sleeping positions)
- 90%+ accuracy on position classification
- Real-time inference on detected poses

### 4. **Full-Stack Integration**
- Python backend for heavy CV/ML processing
- WebSocket for real-time data streaming
- React Native frontend for beautiful mobile UX
- Seamless connection with auto-reconnect

---

## 🎬 Demo Video Analysis

The included **baby_monitor.mp4** video demonstrates:
- ✅ Object detection tracking baby throughout video
- ✅ Safe back sleeping position detection (high confidence)
- ⚠️ Unsafe stomach position detection (high confidence)  
- 📊 Breathing rate: 10-22 BPM range detected
- 🔄 Position transitions: safe ↔ unsafe changes
- 📈 Real-time confidence scores: 0.85-1.00

**Video Stats**: 177 seconds | 5,304 frames | 1920×1080 @ 30 FPS

---

## 🛠️ Tech Stack

### Backend
- **Python 3.12** - Core language
- **MediaPipe** - Object detection and pose estimation
- **OpenCV** - Video processing and frame capture
- **scikit-learn** - ML model training (Random Forest)
- **WebSockets** - Real-time data broadcasting
- **NumPy/SciPy** - Signal processing and numerical operations

### Frontend
- **React Native** - Cross-platform mobile development
- **Expo** - Development framework
- **React Navigation** - Screen navigation
- **WebSocket Client** - Real-time data reception
- **OpenAI GPT-4** - AI-powered recommendations
- **React Native Chart Kit** - Data visualization

---

## 📚 Documentation

### Main Documentation
- **[Frontend README](frontend/README.md)** - Mobile app setup and features
- **[Backend README](backend/README.md)** - Python server setup and algorithms

### Technical Deep Dives
- **[backend/docs/BREATHING_README.md](backend/docs/BREATHING_README.md)** - Breathing detection algorithms
- **[backend/docs/algorithm.md](backend/docs/algorithm.md)** - Mathematical details
- **[backend/docs/validation.md](backend/docs/validation.md)** - Accuracy validation protocol

### Project Information
- **[backend/CHANGELOG.md](backend/CHANGELOG.md)** - Version history
- **[backend/CONTRIBUTING.md](backend/CONTRIBUTING.md)** - Contribution guidelines
- **[LICENSE](LICENSE)** - MIT License

---

## ⚠️ Important Notes

### For Development
**API Keys**: Create a `.env` file in the frontend directory:
```bash
cd frontend
echo "OPENAI_API_KEY=your-key-here" > .env
```

**Python Version**: Use Python 3.11 or 3.12 (MediaPipe doesn't support 3.14+)

### Safety Disclaimer
**FOR RESEARCH/DEMO ONLY - NOT A MEDICAL DEVICE**

This app is NOT:
- ❌ FDA approved
- ❌ Suitable for real medical monitoring
- ❌ A replacement for proper medical equipment

Always use proper medical monitoring devices and follow pediatrician recommendations.

---

## 🏆 Hackathon Information

**Project**: BreatheSafe  
**Category**: AI/ML, Object Detection, Computer Vision, Healthcare  
**Challenge**: Baby sleep safety and SIDS prevention  

**Innovation**:
- Real-time object detection for infant monitoring
- Multi-modal analysis (breathing + position simultaneously)
- Full-stack AI integration (CV backend + mobile frontend)
- Production-ready architecture with error handling

**Impact**: Helps prevent SIDS by detecting unsafe sleep positions and breathing irregularities

---

## 📸 Screenshots & Demo

*(Add screenshots of your app here)*

**Dashboard**: Real-time breathing monitor with position indicators  
**History**: Interactive charts showing breathing patterns  
**Learn**: Educational content about safe sleep practices  
**Settings**: Baby profile and notification preferences

---

## 🚀 Future Enhancements

- [ ] Cloud deployment for remote monitoring
- [ ] Multi-baby support for daycare centers
- [ ] Predictive analytics for sleep pattern analysis
- [ ] Integration with smart nursery devices
- [ ] Clinical validation studies

---

## 📧 Contact

**GitHub**: [tanishajain19](https://github.com/tanishajain19)  
**Repository**: [breathe-safe-sep-hackathon](https://github.com/tanishajain19/breathe-safe-sep-hackathon)

---

**Made with ❤️ for safer baby sleep monitoring**

MIT License | October 2025 | Hackathon Submission
