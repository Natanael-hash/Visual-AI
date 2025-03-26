# AI-Powered HealthCare Application

An AI-driven application designed to assist visually impaired individuals in navigating their surroundings. The app leverages real-time image recognition to enhance accessibility and independence.

![Visual-AI – The augmented eye symbolizes the power of artificial intelligence to perceive and interpret the surrounding environment, providing advanced visual assistance for visually impaired individuals. 🚀](src/web_interface/assets/Visual-AI.png)

## Features

- **Real-Time Image Recognition**: Uses AI to identify objects and obstacles in the user's environment.
- **Accessibility-Focused**: Tailored for visually impaired individuals to improve daily navigation.
- **Powered by AI**: Incorporates advanced machine learning models for accurate detection and processing.

## Tech Stack

- **Programming Language**: Python
- **Object Detection Pre-trained Model**: YOLOv8
- **Depth Estimation Model**: Apple Depth Pro
- **Data Processing**: OpenCV, NumPy, Pandas
- **Deployment**: Streamlit

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Natanael-hash/Visual_AI.git
   cd Visual_AI
   python -m venv env
   # On Mac/Linux:
   source env/bin/activate 
   # On Windows: 
   env\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

### Running the Streamlit Interface

To launch the web application:
```bash
streamlit run app.py
```

This will open the web interface in your default browser, typically at `http://localhost:8501/`.

### Using the Detection Model
 - **Launch the app and point your smartphone camera at the desired surroundings.**
   ```bash
      # The initial script serves as a prototype, where I utilized the model to analyze its accuracy 
      # by applying it to a 4K video.
      from ultralytics import YOLO
      model = YOLO("best.pt")
      model.predict("path/to/video/or/image", save=True, imgsz=640, conf=0.5, device="mps", show=True)
      # This script is designed for Apple Silicon MacBooks that support Metal 
      # for the integrated graphics card. It enables the use of the iPhone's camera 
      # for real-time detection. To utilize the MacBook's webcam, replace the first parameter (0) with 1, and the model will 
      # perform real-time detection from the laptop's camera.
      from ultralytics import YOLO
      model = YOLO("best.pt")
      model.predict(0, save=True, imgsz=640, conf=0.5, device="mps", show=True)
   ```
- **The app will identify and describe objects or obstacles in real time.**
- **Enjoy enhanced accessibility and navigation!**

## Future Improvements
- **Enhance object detection accuracy with more diverse datasets.**
- **Integrate voice guidance for real-time audio feedback.**
- **Expand compatibility to other platforms (e.g., iOS, Android).**

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your improvements or ideas.

## Contact
- **Author: Natanael Hordon**
- **Email: natanaelhordon@icloud.com**
- **GitHub: https://github.com/Natanael-hash**
- **LinkedIn: http://linkedin.com/in/natanael-hordon-b04bb22b5**