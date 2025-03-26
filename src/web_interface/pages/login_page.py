import streamlit as st

def show_login_page():
    t = {
        "Română": {
            "title": "🔑 Autentificare",
            "username": "👤 Utilizator",
            "password": "🔑 Parolă",
            "login_button": "Conectează-te",
            "forgot_password": "🔑 Ai uitat parola?",
            "create_account": "📝 Nu ai cont? Creează unul",
            "fill_fields": "⚠️ Completați toate câmpurile.",
            "invalid_credentials": "❌ Nume de utilizator sau parolă incorecte.",
        },
        "English": {
            "title": "🔑 Login",
            "username": "👤 Username",
            "password": "🔑 Password",
            "login_button": "Log In",
            "forgot_password": "🔑 Forgot Password?",
            "create_account": "📝 Don’t have an account? Sign Up",
            "fill_fields": "⚠️ Please fill in all fields.",
            "invalid_credentials": "❌ Invalid username or password.",
        }
    }

    language = st.session_state.language
    st.title(t[language]["title"])

    with st.form("login_form"):
        username = st.text_input(t[language]["username"], placeholder=t[language]["username"])
        password = st.text_input(t[language]["password"], type="password", placeholder=t[language]["password"])
        submit_button = st.form_submit_button(t[language]["login_button"])

    if submit_button:
        if not username or not password:
            st.error(t[language]["fill_fields"])
        elif username == "admin" and password == "password":
            st.success(f"✅ Bine ai venit, {username}!")
        else:
            st.error(t[language]["invalid_credentials"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button(t[language]["forgot_password"]):
            st.session_state.page = "forgot_password"
            st.rerun()
    with col2:
        if st.button(t[language]["create_account"]):
            st.session_state.page = "signup"
            st.rerun()