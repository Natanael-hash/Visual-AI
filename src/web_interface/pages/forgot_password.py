import streamlit as st


def show_forgot_password_page():
    t = {
        "Română": {
            "title": "🔑 Recuperare Parolă",
            "email": "📧 Adresa de Email",
            "instructions": "Introdu adresa de email asociată contului tău. Îți vom trimite un link pentru resetarea parolei.",
            "reset_button": "Trimite Linkul de Resetare",
            "back_to_login": "↩️ Înapoi la Autentificare",
            "fill_email": "⚠️ Te rugăm să introduci adresa de email.",
            "success": "✅ Un email cu instrucțiuni pentru resetarea parolei a fost trimis la adresa ta de email (simulat)."
        },
        "English": {
            "title": "🔑 Password Recovery",
            "email": "📧 Email Address",
            "instructions": "Enter the email address associated with your account. We will send you a password reset link.",
            "reset_button": "Send Reset Link",
            "back_to_login": "↩️ Back to Login",
            "fill_email": "⚠️ Please enter your email address.",
            "success": "✅ An email with password reset instructions has been sent to your email address (simulated)."
        }
    }

    language = st.session_state.language
    st.title(t[language]["title"])
    st.markdown(t[language]["instructions"])

    with st.form("forgot_password_form"):
        email = st.text_input(t[language]["email"], placeholder=t[language]["email"])
        submit_button = st.form_submit_button(t[language]["reset_button"])

    if submit_button:
        if not email:
            st.error(t[language]["fill_email"])
        else:
            st.success(t[language]["success"])
            # În aplicația reală, aici ai trimite un email cu link de resetare

    if st.button(t[language]["back_to_login"]):
        st.session_state.page = "login"
        st.rerun()