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
    page_title="CaptureMate Pro",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

[Complete file content from source]