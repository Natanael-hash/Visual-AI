import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from .yolo_predictions_and_distance_estimation import ObjectDetection_and_Distance_Estimation

@st.cache_resource
def load_detector():
    return ObjectDetection_and_Distance_Estimation(capture_index=0)

def run_visual_ai():
    st.title("Real-Time Object Detection & Depth Estimation with Voice Feedback")

    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False

    if st.button("Start Camera 🎥"):
        st.session_state.camera_active = True

    if st.session_state.camera_active:
        detector = load_detector()

        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            results = detector.predict(img)
            annotated_frame, object_boxes, depth = detector.plot_bboxes_and_depth_estimation(results, img)
            detector.get_directions(img, object_boxes, depth)
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

        webrtc_streamer(key="yolo-depth-voice", video_frame_callback=video_frame_callback)

    if st.button("Reset Camera Session 🔄"):
        st.session_state.camera_active = False
        st.rerun()