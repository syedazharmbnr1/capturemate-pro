import streamlit as st
import os
import time
import datetime
import cv2
import json
import numpy as np
import pyautogui
import mss
import zipfile
import tempfile

from pynput import mouse, keyboard
from PIL import Image, ImageFilter
import pyaudio
import wave

import moviepy.editor as mpe
from moviepy.editor import VideoFileClip, vfx
import ffmpeg  # ffmpeg-python
import threading
import pyautogui

# --------------------------------------------------------------------------
#  STYLING & PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Professional Screen Recorder",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Modern Light Theme */
    .stApp {
        background-color: #FFFFFF;
        color: #333333;
    }
    
    /* Header Styling */
    h1, h2, h3 {
        color: #1a237e;
        font-weight: 600;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(45deg, #2196F3, #1976D2);
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        border: none;
        font-weight: 500;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Input Fields */
    .stSelectbox, .stNumberInput {
        background-color: #F5F5F5;
        border-radius: 8px;
        border: 1px solid #E0E0E0;
        padding: 8px;
    }
    
    /* Tabs Styling */
    .stTabs {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background-color: #2196F3;
    }
    
    /* Card-like containers */
    .css-1r6slb0 {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

[Rest of the code...]