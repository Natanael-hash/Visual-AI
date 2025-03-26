import streamlit as st

def show_signup_page():
    t = {
        "Română": {
            "title": "📝 Înregistrare",
            "username": "👤 Nume utilizator",
            "email": "📧 Email",
            "password": "🔑 Parolă",
            "confirm_password": "🔐 Confirmă parola",
            "signup_button": "Creează cont",
            "already_have_account": "🔑 Ai deja un cont? Conectează-te",
            "fill_fields": "⚠️ Completați toate câmpurile.",
            "password_mismatch": "❌ Parolele nu coincid!",
            "success": "✅ Cont creat cu succes! Acum te poți conecta."
        },
        "English": {
            "title": "📝 Sign Up",
            "username": "👤 Username",
            "email": "📧 Email",
            "password": "🔑 Password",
            "confirm_password": "🔐 Confirm Password",
            "signup_button": "Sign Up",
            "already_have_account": "🔑 Already have an account? Log In",
            "fill_fields": "⚠️ Please fill in all fields.",
            "password_mismatch": "❌ Passwords do not match!",
            "success": "✅ Account successfully created! You can now log in."
        }
    }

    language = st.session_state.language
    st.title(t[language]["title"])

    with st.form("signup_form"):
        username = st.text_input(t[language]["username"], placeholder=t[language]["username"])
        email = st.text_input(t[language]["email"], placeholder=t[language]["email"])
        password = st.text_input(t[language]["password"], type="password", placeholder=t[language]["password"])
        confirm_password = st.text_input(t[language]["confirm_password"], type="password", placeholder=t[language]["confirm_password"])
        submit_button = st.form_submit_button(t[language]["signup_button"])

    if submit_button:
        if not username or not email or not password or not confirm_password:
            st.error(t[language]["fill_fields"])
        elif password != confirm_password:
            st.error(t[language]["password_mismatch"])
        else:
            st.success(t[language]["success"])
            st.session_state.page = "login"

    if st.button(t[language]["already_have_account"]):
        st.session_state.page = "login"
        st.rerun()