# Visual AI: AI-Powered HealthCare Application

An AI-driven application designed to assist visually impaired individuals in navigating their surroundings. The app leverages real-time image recognition to enhance accessibility and independence.

![Visual-AI – The augmented eye symbolizes the power of artificial intelligence to perceive and interpret the surrounding environment, providing advanced visual assistance for visually impaired individuals. 🚀](src/web_interface/assets/Visual-AI.png)

---

## 🔍 Features

- **Real-Time Object Detection**: Identifies obstacles and important objects around the user using YOLOv8.
- **Depth Estimation**: Estimates the distance between user and object using Apple's Depth Pro model.
- **Voice Feedback System**: Guides the user with audio cues, helping them navigate safely.
- **Streamlit-Based Interface**: Clean and interactive UI for quick testing and usability.
- **Designed for Accessibility**: Tailored for real-world assistive scenarios.

---

## 🧠 Tech Stack

| Component            | Description                                        |
|---------------------|----------------------------------------------------|
| Language             | Python                                             |
| Object Detection     | YOLOv8 (Ultralytics)                               |
| Depth Estimation     | Apple Depth Pro                          |
| Data Processing      | OpenCV, NumPy, Pandas                              |
| Voice Feedback       | pyttsx3                                            |
| Web Interface        | Streamlit                                          |
| Deployment Target    | AWS (with RDS for DB support)                     |

---

## 🚀 Getting Started

### 1. Clone & Setup
```bash
git clone https://github.com/Natanael-hash/Visual-AI.git
cd Visual_AI
python -m venv env
source env/bin/activate  # On Mac/Linux
# or
env\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 2. Run Streamlit Interface
```bash
streamlit run app.py
```

Visit `http://localhost:8501` to interact with the interface.

---

## 🎥 Object Detection (Prototype)

For testing with 4K video or webcam:

```python
from ultralytics import YOLO
model = YOLO("object_detection.pt")
# Test with video or image
model.predict("path/to/video/or/image", save=True, imgsz=640, conf=0.5, device="mps", show=True)
# Real-time webcam detection
model.predict(0, save=True, imgsz=640, conf=0.5, device="mps", show=True)
```

---

## 📈 Future Roadmap

- [ ] Improve detection for edge-cases (e.g., low light).
- [ ] Add multilingual voice support.
- [ ] Expand platform to Android/iOS.
- [ ] Add user sign-up/login with MySQL and RDS integration.

---

## 🤝 Contributing

Feel free to fork and contribute — PRs are welcome!

---

## 📫 Contact

**Author**: Natanael Hordon  
📧 Email: natanaelhordon@icloud.com  
🔗 GitHub: [Natanael-hash](https://github.com/Natanael-hash)  
🔗 LinkedIn: [natanael-hordon-b04bb22b5](https://linkedin.com/in/natanael-hordon-b04bb22b5)
