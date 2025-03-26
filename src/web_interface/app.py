import streamlit as st
from pages import about_me
from pages import login_page
from pages import sign_up
from pages import forgot_password

# st.markdown("""
#     <style>
#     [data-testid="stSidebarNav"] {display: none;}
#     </style>
# """, unsafe_allow_html=True)

st.set_page_config(page_title="Visual-AI", page_icon="🤖", layout="wide")

if "high_contrast" not in st.session_state:
    st.session_state.high_contrast = False
if "language" not in st.session_state:
    st.session_state.language = "Română"

def apply_styles():
    bg_color = "#000000" if st.session_state.high_contrast else "#000000"
    text_color = "#FFD700" if st.session_state.high_contrast else "#FFF"
    header_color = "#FF500" if st.session_state.high_contrast else "#4CAF50"

    st.markdown(f"""
        <style>
        html, body, [data-testid="stAppViewContainer"] {{
            background-color: {bg_color} !important;
            color: {text_color} !important;
        }}
        body {{
            background-color: {bg_color} !important;
            color: {text_color} !important;
            font-family: Arial, sans-serif;
        }}
        .title-text {{
            font-size: 32px;
            font-weight: bold;
            text-align: center;
            color: {header_color} !important;
        }}
        .section-title {{
            font-size: 24px;
            font-weight: bold;
            margin-top: 20px;
            color: {header_color} !important;
        }}
        .section-content {{
            font-size: 18px;
            line-height: 1.6;
        }}
        </style>
    """, unsafe_allow_html=True)

apply_styles()

translations = {
    "Română": {
        "home_title": "🏠 Acasă",
        "about": "👤 Despre Mine",
        "login": "🔑 Autentificare",
        "language": "🌐 Schimbă limba",
        "contrast_button": "🎨 High-Contrast",
        "welcome_message": "Visual AI – Asistent Inteligent pentru Persoane cu Deficiențe de Vedere",
        "app_description": """
        ## 🔍 Descriere Generală
        Visual AI este o aplicație inovatoare dezvoltată pentru a ajuta persoanele cu deficiențe de vedere să navigheze în siguranță și să interacționeze mai ușor cu mediul înconjurător.  
        Folosind inteligența artificială și cele mai noi tehnologii din domeniul computer vision, aplicația oferă descrieri audio în timp real ale obiectelor și obstacolelor din jurul utilizatorului.  
        În prima fază, aplicația este disponibilă ca aplicație web dezvoltată în Python cu Streamlit, iar ulterior va fi lansată și ca aplicație nativă pentru iOS, utilizând Swift și Core ML pentru inferență optimizată pe dispozitive Apple.

        ## 🛠️ Funcționalități Principale  
        - **🔹 Detecția Obiectelor (Object Detection)**  
          - Model **YOLOv8** antrenat pe un dataset personalizat cu **30.000+ imagini** și **36 de clase**.  
          - Feedback audio instantaneu pentru recunoaștere rapidă.  
          - Adaptabilitate pentru diferite medii (exterior, interior, spații aglomerate).  

        - **🔹 Estimarea Distanței cu Tehnologia LiDAR**  
          - Integrare cu **Apple ml-depth-pro** pentru măsurarea distanței.  
          - Detectarea adâncimii și generarea de feedback audio personalizat.  

        - **🔹 Feedback Audio Inteligent**  
          - Descriere vocală clară a obiectelor detectate prin **Text-to-Speech (TTS)**.  
          - Adaptare a tonului și volumului în funcție de distanță.  
          - Suport pentru **comenzi vocale** (în dezvoltare).  

        - **🔹 Interfață Adaptată pentru Accesibilitate**  
          - Design minimalist, optimizat pentru utilizatori cu deficiențe de vedere.  
          - **Suport pentru mod Light/Dark** pentru vizibilitate maximă.  

        - **🔹 Mod Real-Time & Suport pentru Video**  
          - Optimizare pentru procesare rapidă a imaginilor și videoclipurilor.  
          - Algoritmi adaptați pentru performanță ridicată și consum redus de resurse.  
          
        ## 🛠️ Tehnologii Utilizate
        ✅ **Python + Streamlit** (pentru prototipul web)  
        ✅ **YOLOv8 + ONNX + Core ML** (pentru inferența modelului)  
        ✅ **Apple ml-depth-pro + LiDAR** (pentru estimarea distanței)  
        ✅ **Swift + SwiftUI** (pentru aplicația iOS)  
        ✅ **Text-to-Speech (TTS) & Speech Recognition** (pentru interacțiunea audio)  
        ✅ **FastAPI + SQLAlchemy** (pentru backend-ul aplicației)

        ## 🔮 Planuri de Viitor  
        - 🏁 Navigație avansată cu ghidare bazată pe analiza mediului.  
        - 📍 Integrare cu **Apple Maps** și GPS pentru ghidare vocală.  
        - 🔄 Extinderea aplicației pentru **Android** cu **TensorFlow Lite**.  
        - 🌍 Suport multi-limbă (Română, Engleză și altele). 
        
        ## 🏁 Concluzie
        Visual AI este un pas important în direcția accesibilității inteligente, combinând computer vision, machine learning și tehnologii de ultimă generație pentru a oferi o soluție completă și inovatoare pentru persoanele cu deficiențe de vedere.Scopul final este de a oferi mai multă autonomie și siguranță utilizatorilor, ajutându-i să navigheze și să interacționeze cu lumea din jur în mod intuitiv și eficient. 
        """,
    },
    "English": {
        "home_title": "🏠 Home",
        "about": "👤 About Me",
        "login": "🔑 Login",
        "language": "🌐 Change Language",
        "contrast_button": "🎨 High-Contrast",
        "welcome_message": "Visual AI – Intelligent Assistant for Visually Impaired People",
        "app_description": """
        ## 🔍 General Overview
        Visual AI is an innovative application developed to help visually impaired individuals navigate safely and interact more easily with their surroundings.  
        Using artificial intelligence and the latest technologies in computer vision, the app provides real-time audio descriptions of objects and obstacles around the user.  
        In the first phase, the application is available as a web application developed in Python with Streamlit, and later it will be launched as a native iOS application using Swift and Core ML for optimized inference on Apple devices.

        ## 🛠️ Key Features  
        - **🔹 Object Detection**  
          - **YOLOv8 model** trained on a custom dataset with **30,000+ images** and **36 object classes**.  
          - Instant audio feedback for real-time object recognition.  
          - Adaptability for different environments (outdoor, indoor, crowded spaces).  

        - **🔹 Distance Estimation with LiDAR**  
          - Integration with **Apple ml-depth-pro** for precise depth measurement.  
          - Depth detection and customized audio feedback.  

        - **🔹 Smart Audio Feedback**  
          - Clear **Text-to-Speech (TTS)** description of detected objects.  
          - Volume and tone adaptation based on distance.  
          - **Voice commands** support (upcoming feature).  

        - **🔹 Accessibility-Oriented Interface**  
          - Minimalist design optimized for visually impaired users.  
          - **Light/Dark Mode support** for maximum visibility.  

        - **🔹 Real-Time Mode & Video Processing**  
          - Optimized for fast image and video processing.  
          - Algorithms adjusted for high performance and low resource consumption.
          
        ## 🛠️ Technologies Used
        ✅ **Python + Streamlit** (for the web prototype)  
        ✅ **YOLOv8 + ONNX + Core ML** (for model inference)  
        ✅ **Apple ml-depth-pro + LiDAR** (for distance estimation)  
        ✅ **Swift + SwiftUI** (for the iOS app)  
        ✅ **Text-to-Speech (TTS) & Speech Recognition** (for audio interaction)  
        ✅ **FastAPI + SQLAlchemy** (for the app backend)    

        ## 🔮 Future Plans  
        - 🏁 Advanced navigation with AI-based guidance.  
        - 📍 **Apple Maps** and GPS integration for voice-guided navigation.  
        - 🔄 Expanding to **Android** with **TensorFlow Lite**.  
        - 🌍 Multi-language support (Romanian, English, and more).
        
        ## 🏁 Conclusion
        Visual AI is an important step towards intelligent accessibility, combining computer vision, machine learning, and cutting-edge technologies to provide a complete and innovative solution for visually impaired individuals.The ultimate goal is to offer users greater autonomy and safety, helping them navigate and interact with the world around them in an intuitive and efficient manner. 
        
        
        """

        ,
    },
}

if "language" not in st.session_state:
    st.session_state.language = "Română"

language = st.sidebar.selectbox(
    "🌍 Selectează limba / Choose Language:", ["Română", "English"],
    index=0 if st.session_state.language == "Română" else 1
)
st.session_state.language = language
t = translations[language]


if "page" not in st.session_state:
    st.session_state.page = "home"

st.sidebar.title("🔎 Navigare")
nav_options = [t["home_title"], t["about"], t["login"]]
page = st.sidebar.radio("", nav_options)

if st.sidebar.button(t["contrast_button"]):
    st.session_state.high_contrast = not st.session_state.high_contrast
    apply_styles()
    st.rerun()

if st.session_state.page == "signup":
    sign_up.show_signup_page()
elif st.session_state.page == "forgot_password":
    forgot_password.show_forgot_password_page()
else:
    if page == t["home_title"]:
        st.session_state.page = "home"
        st.title(t["welcome_message"])
        st.markdown(t["app_description"])
    elif page == t["about"]:
        st.session_state.page = "about"
        about_me.show_about_page()
    elif page == t["login"]:
        st.session_state.page = "login"
        login_page.show_login_page()