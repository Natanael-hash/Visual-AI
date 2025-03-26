import streamlit as st

def show_about_page():
    t = {
        "Română": {
            "title": "👤 Despre mine",
            "description": """
            Salut! Mă numesc **Hordon Natanael** și sunt student la Facultatea de Inginerie a Sistemelor la **Aurel Vlaicu**.  
            Sunt pasionat de tehnologie și dezvolt aplicația **Visual-AI**, un proiect inovator pentru a ajuta persoanele cu deficiențe de vedere.
            """,
            "project_title": "🚀 Despre Visual-AI",
            "project_description": """
            **Visual-AI** este o aplicație ce asistă persoanele cu deficiențe de vedere în navigarea sigură în mediul înconjurător.  
            Aceasta folosește **inteligența artificială** pentru:  
            - Detectarea obiectelor  
            - Aprecierea precisă a distanței dintre obstacole  
            - Ghidare prin răspuns vocal  

            **Implementări viitoare:**  
            ✅ Aplicație **iOS** cu senzor **LiDAR** pentru precizie ridicată  
            ✅ Integrare cu **Apple Maps** pentru ghidare către stații de autobuz, magazine etc.  
            """,
            "contact": "📩 Contact",
            "email": "Email: natanael.hordon@icloud.com",
        },
        "English": {
            "title": "👤 About Me",
            "description": """
            Hello! My name is **Hordon Natanael**, and I am a student at the Faculty of Systems Engineering at **Aurel Vlaicu** University.  
            I am passionate about technology and developing **Visual-AI**, an innovative project designed to assist visually impaired individuals.
            """,
            "project_title": "🚀 About Visual-AI",
            "project_description": """
            **Visual-AI** is an application that helps visually impaired individuals navigate safely in their surroundings.  
            It utilizes **artificial intelligence** to:  
            - Detect objects  
            - Accurately estimate distances  
            - Provide voice guidance  

            **Future Implementations:**  
            ✅ **iOS app** with **LiDAR sensor** for high precision  
            ✅ **Apple Maps integration** for navigation to bus stations, stores, etc.  
            """,
            "contact": "📩 Contact",
            "email": "Email: natanael.hordon@icloud.com",
        }
    }

    language = st.session_state.language
    st.title(t[language]["title"])
    st.image("assets/6B84E397-8F2C-4963-8376-411A019E266D_1_102_a.jpeg", caption="Hordon Natanael", use_container_width=False, width=300)
    st.markdown(t[language]["description"])
    st.header(t[language]["project_title"])
    st.markdown(t[language]["project_description"])
    st.header(t[language]["contact"])
    st.markdown(f"📧 **{t[language]['email']}**")