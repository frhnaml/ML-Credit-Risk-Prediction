# src/utils/load_css.py
import streamlit as st
import os

def load_css(file_name="assets/style.css"):
    """
    Load custom CSS ke halaman Streamlit
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    css_path = os.path.join(base_dir, file_name)
    
    try:
        with open(css_path, "r") as f:
            css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found: {css_path}")
