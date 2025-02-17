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

# --------------------------------------------------------------------------
#  RECORDING FUNCTIONS
# --------------------------------------------------------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def zip_project_folder(folder_path):
    zip_name = os.path.basename(folder_path) + ".zip"
    zip_path = os.path.join(os.path.dirname(folder_path), zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, folder_path)
                zf.write(full_path, arcname=rel_path)
    return zip_path

def record_microphone_audio(duration_sec=5, output_wav="mic_audio.wav", channels=1, rate=44100, chunk=1024):
    """
    Improved audio recording with better sync
    """
    try:
        p = pyaudio.PyAudio()
        
        # Add input device validation
        device_info = p.get_default_input_device_info()
        if not device_info:
            raise Exception("No input device found")
            
        # Improved stream configuration
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=chunk,
            input_device_index=device_info['index']
        )
        
        # Use precise timing
        frames = []
        start_time = time.time()
        
        while (time.time() - start_time) < duration_sec:
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
            
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save with proper format
        with wave.open(output_wav, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
            
    except Exception as e:
        st.error(f"Audio recording error: {str(e)}")
        return None

# --------------------------------------------------------------------------
#  USER-CHOSEN WEBCAM POSITION
# --------------------------------------------------------------------------
def place_webcam_overlay(frame_screen, frame_cam, position, offset=10):
    """
    Places the webcam overlay in a chosen corner: top-left, top-right,
    bottom-left, or bottom-right. 'offset' is the margin from edges.
    """
    cam_h, cam_w, _ = frame_cam.shape
    screen_h, screen_w, _ = frame_screen.shape

    if position == "top-left":
        x1, y1 = offset, offset
    elif position == "top-right":
        x1 = screen_w - cam_w - offset
        y1 = offset
    elif position == "bottom-left":
        x1 = offset
        y1 = screen_h - cam_h - offset
    else:  # "bottom-right"
        x1 = screen_w - cam_w - offset
        y1 = screen_h - cam_h - offset

    x2 = x1 + cam_w
    y2 = y1 + cam_h

    # Ensure it doesn't exceed the screen
    if x1 < 0 or y1 < 0 or x2 > screen_w or y2 > screen_h:
        return frame_screen  # skip if out of bounds

    frame_screen[y1:y2, x1:x2] = frame_cam
    return frame_screen

def record_screen_webcam(
    duration_sec=5,
    fps=30,
    record_webcam=False,
    webcam_index=0,
    webcam_position="bottom-right",
    webcam_size=25,
    record_audio=False,
    audio_sources=None,
    quality="high",
    cursor_highlight=False,
    highlight_color=None,
    highlight_size=20,
    click_effects=False,
    click_color=None,
    click_animation=None,
    selected_screen="Main Screen",
    encoding="H.264",
    bitrate="8 Mbps",
    project_root="recordings",
    screen_region=None,
    hide_desktop_icons=False
):
    """
    Records screen + optional webcam overlay with enhanced features
    """
    # Initialize audio sources list
    if audio_sources is None:
        audio_sources = []

    # Create project folder
    ensure_dir(project_root)
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    project_folder = os.path.join(project_root, f"record_{now_str}")
    ensure_dir(project_folder)

    # Set up screen capture
    frames_bgr = []
    sct = mss.mss()
    
    # Handle screen selection
    if selected_screen == "Custom Region" and screen_region:
        monitor = screen_region
    else:
        monitor = sct.monitors[1]  # Default to main screen
    
    screen_w = monitor["width"]
    screen_h = monitor["height"]

    # Set up webcam if enabled
    cap_webcam = None
    if record_webcam:
        cap_webcam = cv2.VideoCapture(webcam_index)
        if not cap_webcam.isOpened():
            st.warning("Failed to open webcam. Proceeding without webcam.")
            cap_webcam = None

    # Set up mouse and keyboard listeners
    click_events = []
    keystroke_events = []
    start_time = time.time()

    def on_click(x, y, button, pressed):
        if pressed:
            t = time.time() - start_time
            click_events.append((t, x, y, str(button)))
            
    def on_press(key):
        t = time.time() - start_time
        try:
            key_str = key.char if key.char else str(key)
        except:
            key_str = str(key)
        keystroke_events.append((t, key_str))

    # Start listeners
    mouse_listener = mouse.Listener(on_click=on_click)
    keyboard_listener = keyboard.Listener(on_press=on_press)
    mouse_listener.start()
    keyboard_listener.start()

    # Start audio recording if enabled
    audio_thread = None
    if record_audio and "Microphone" in audio_sources:
        audio_path = os.path.join(project_folder, "mic_audio.wav")
        audio_thread = threading.Thread(
            target=record_microphone_audio,
            args=(duration_sec, audio_path)
        )
        audio_thread.start()

    # Create cursor effect settings dictionary
    cursor_settings = {
        'enabled': cursor_highlight,
        'color': highlight_color or "#FFD700",
        'size': highlight_size,
        'opacity': 0.3,  # Default opacity
        'click_effects': click_effects,
        'click_color': click_color or "#FF0000",
        'click_animation': click_animation or "circle"
    }

    # Main recording loop
    frame_interval = 1.0 / fps
    last_frame_time = time.time()
    current_cursor_pos = pyautogui.position()

    while True:
        elapsed = time.time() - start_time
        if elapsed > duration_sec:
            break

        now = time.time()
        if now - last_frame_time >= frame_interval:
            # Capture screen
            img = sct.grab(monitor)
            frame_screen = np.array(img, dtype=np.uint8)
            frame_screen = cv2.cvtColor(frame_screen, cv2.COLOR_BGRA2BGR)

            # Add cursor effects if enabled
            if cursor_settings['enabled']:
                cursor_pos = pyautogui.position()
                
                # Smooth cursor movement
                current_cursor_pos = smooth_cursor_movement(
                    current_cursor_pos,
                    cursor_pos
                )
                
                frame_screen = apply_cursor_effects(
                    frame_screen,
                    current_cursor_pos,
                    clicked=bool(click_events and abs(click_events[-1][0] - elapsed) < 0.1),
                    style=cursor_settings['click_animation'],
                    color=cursor_settings['color'],
                    opacity=cursor_settings['opacity']
                )

            # Add webcam overlay if enabled
            if cap_webcam:
                ret_cam, frame_cam = cap_webcam.read()
                if ret_cam:
                    frame_cam = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2RGB)
                    frame_cam = cv2.cvtColor(frame_cam, cv2.COLOR_RGB2BGR)

                    # Scale webcam based on size parameter
                    scale_factor = webcam_size / 100.0
                    new_w = int(frame_cam.shape[1] * scale_factor)
                    new_h = int(frame_cam.shape[0] * scale_factor)
                    frame_cam_small = cv2.resize(frame_cam, (new_w, new_h))

                    frame_screen = place_webcam_overlay(
                        frame_screen,
                        frame_cam_small,
                        webcam_position
                    )

            frames_bgr.append(frame_screen)
            last_frame_time = now

    # Cleanup
    mouse_listener.stop()
    keyboard_listener.stop()
    if cap_webcam:
        cap_webcam.release()
    if audio_thread:
        audio_thread.join()

    # Save video and project data
    if not frames_bgr:
        raise RuntimeError("No frames captured!")

    out_video_path = os.path.join(project_folder, "recording.mp4")
    h, w, _ = frames_bgr[0].shape
    
    # Use codec based on encoding parameter
    if encoding == "H.264":
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    elif encoding == "H.265":
        fourcc = cv2.VideoWriter_fourcc(*'hev1')
    else:  # Default to H.264
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    
    out_writer = cv2.VideoWriter(
        out_video_path, 
        fourcc, 
        fps, 
        (w, h),
        True
    )
    
    for f in frames_bgr:
        out_writer.write(f)
    out_writer.release()

    # Save project data
    project_data = {
        "json": {
            "scenes": [
                {
                    "name": "Default",
                    "zoomRanges": [
                        {
                            "startTime": int(t_sec * 1000),
                            "endTime": int(t_sec * 1000) + 1000,
                            "zoom": 2,
                            "manualTargetPoint": {
                                "x": x / w,
                                "y": y / h
                            },
                            "isDisabled": False
                        }
                        for (t_sec, x, y, btn) in click_events
                    ]
                }
            ],
            "keystrokes": keystroke_events,
            "settings": {
                "quality": quality,
                "encoding": encoding,
                "bitrate": bitrate,
                "cursor_highlight": cursor_highlight,
                "highlight_color": highlight_color,
                "highlight_size": highlight_size,
                "click_effects": click_effects,
                "click_animation": click_animation,
                "cursor_settings": cursor_settings
            }
        }
    }
    
    project_json_path = os.path.join(project_folder, "project.json")
    with open(project_json_path, "w", encoding="utf-8") as f:
        json.dump(project_data, f, indent=2)

    return project_folder

# --------------------------------------------------------------------------
#  MOTION BLUR & ZOOM
# --------------------------------------------------------------------------

def apply_motion_blur(frame_bgr, blur_radius=5):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    pil_img = pil_img.filter(ImageFilter.BoxBlur(blur_radius))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def get_active_zoom_range(zoom_ranges, time_sec):
    for zr in zoom_ranges:
        start_sec = zr.get("startTime", 0) / 1000.0
        end_sec = zr.get("endTime", 0) / 1000.0
        if not zr.get("isDisabled", False) and start_sec <= time_sec <= end_sec:
            return zr
    return None

def compute_zoomed_frame(frame_bgr, zoom_range, webcam_frame=None, webcam_pos=None):
    """
    Apply zoom while preserving webcam overlay - fixed version
    """
    # Store original frame
    original_frame = frame_bgr.copy()
    h, w, _ = frame_bgr.shape
    
    # Apply zoom to main content only
    zoom_factor = zoom_range.get("zoom", 1)
    if zoom_factor <= 1.0:
        return frame_bgr
        
    # Calculate zoom area
    cx = zoom_range.get("manualTargetPoint", {}).get("x", 0.5) * w
    cy = zoom_range.get("manualTargetPoint", {}).get("y", 0.5) * h
    
    crop_w = w / zoom_factor
    crop_h = h / zoom_factor
    
    x1 = int(cx - crop_w / 2)
    y1 = int(cy - crop_h / 2)
    x2 = int(cx + crop_w / 2)
    y2 = int(cy + crop_h / 2)
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    # Zoom main content
    cropped = frame_bgr[y1:y2, x1:x2]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Restore webcam region from original frame
    if webcam_frame is not None and webcam_pos is not None:
        wh, ww, _ = webcam_frame.shape
        if webcam_pos == "top-left":
            zoomed[0:wh, 0:ww] = original_frame[0:wh, 0:ww]
        elif webcam_pos == "top-right":
            zoomed[0:wh, w-ww:w] = original_frame[0:wh, w-ww:w]
        elif webcam_pos == "bottom-left":
            zoomed[h-wh:h, 0:ww] = original_frame[h-wh:h, 0:ww]
        else:  # bottom-right
            zoomed[h-wh:h, w-ww:w] = original_frame[h-wh:h, w-ww:w]
            
    return zoomed

def overlay_keystrokes(frame_bgr, keystrokes, current_sec, style="Modern", position="Bottom"):
    recent_keys = [k for (t, k) in keystrokes if current_sec - 1 <= t <= current_sec]
    if not recent_keys:
        return frame_bgr
        
    text = " + ".join(recent_keys)
    out = frame_bgr.copy()
    
    # Get frame dimensions
    h, w = frame_bgr.shape[:2]
    
    # Configure text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    text_size = cv2.getTextSize(f"Keys: {text}", font, font_scale, thickness)[0]
    
    # Calculate position based on setting
    padding = 10
    if position == "Bottom":
        start_x = 50
        start_y = h - 50
    elif position == "Top":
        start_x = 50
        start_y = text_size[1] + 30
    elif position == "Left":
        start_x = 30
        start_y = h // 2
    else:  # Right
        start_x = w - text_size[0] - 30
        start_y = h // 2
    
    if style == "Modern":
        # Modern style with background box
        text_size = cv2.getTextSize(f"Keys: {text}", font, font_scale, thickness)[0]
        
        # Draw background box
        cv2.rectangle(
            out,
            (start_x - padding, start_y - text_size[1] - padding),
            (start_x + text_size[0] + padding, start_y + padding),
            (0, 0, 0, 0.7),
            -1
        )
        
        # Draw text
        cv2.putText(
            out,
            f"Keys: {text}",
            (start_x, start_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )
    
    elif style == "Classic":
        # Simple text overlay
        cv2.putText(
            out,
            f"Keys: {text}",
            (start_x, start_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
    
    elif style == "Minimal":
        # Just the keys without "Keys:" prefix
        cv2.putText(
            out,
            text,
            (start_x, start_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
    
    return out

# --------------------------------------------------------------------------
#  RE-ENCODE WITH LIBX264 (QuickTime-friendly)
# --------------------------------------------------------------------------

def reencode_with_libx264(in_path, out_path):
    """
    Re-encode with libx264 for QuickTime compatibility
    """
    try:
        # Verify input file exists and has content
        if not os.path.exists(in_path):
            st.error("Input video file not found")
            return False
            
        if os.path.getsize(in_path) == 0:
            st.error("Input video file is empty")
            return False

        # Build the pipeline with explicit input/output parameters
        try:
            # Read input video stream
            probe = ffmpeg.probe(in_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            width = int(video_info['width'])
            height = int(video_info['height'])
            
            # Create the ffmpeg pipeline with explicit parameters
            stream = (
                ffmpeg
                .input(in_path)
                .output(
                    out_path,
                    vcodec='libx264',
                    acodec='aac',
                    video_bitrate='4000k',
                    audio_bitrate='192k',
                    r=30,  # frame rate
                    pix_fmt='yuv420p',
                    preset='ultrafast',
                    movflags='+faststart',
                    f='mp4',  # force MP4 format
                    s=f'{width}x{height}'  # maintain original dimensions
                )
                .overwrite_output()
            )
            
            # Run FFmpeg with error handling
            try:
                stream.run(capture_stdout=True, capture_stderr=True)
                
                # Verify output file was created
                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    return True
                else:
                    st.error("Failed to create output file")
                    return False
                    
            except ffmpeg.Error as e:
                error_message = e.stderr.decode() if e.stderr else str(e)
                st.error(f"FFmpeg encoding error: {error_message}")
                return False
                
        except Exception as e:
            st.error(f"FFmpeg pipeline error: {str(e)}")
            return False
            
    except Exception as e:
        st.error(f"Unexpected error during re-encoding: {str(e)}")
        return False

# --------------------------------------------------------------------------
#  PLAYBACK & EDITING
# --------------------------------------------------------------------------

def play_with_zoom(
    project_folder,
    apply_motion_blur_flag=False,
    blur_intensity=2,
    overlay_keys_flag=False,
    key_style="Modern",
    vertical_mode=False,
    export_gif=False,
    cut_start_sec=0,
    cut_end_sec=None,
    speed_factor=1.0,
    zoom_intensity=2.0,
    zoom_smoothness=0.5,
    key_position="center",
    background_settings=None,
    show_mouse=False,
    click_style=None,
    click_color=None,
    click_opacity=0.7,
    gif_quality="Medium",
    preview_quality="Medium"
):
    """Enhanced video playback with zoom, effects, and webcam preservation"""
    
    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    frames_written = 0
    frames_for_gif = []
    current_frame_idx = 0

    # Setup paths
    video_path = os.path.join(project_folder, "recording.mp4")
    json_path = os.path.join(project_folder, "project.json")
    
    # Validation
    if not os.path.exists(project_folder):
        st.error("Project folder not found.")
        return None
        
    if not os.path.isfile(video_path):
        st.error("Video file not found. Please record first.")
        return None

    try:
        # Load project data
        with open(json_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
            settings = data.get("json", {}).get("settings", {})
            scenes = data.get("json", {}).get("scenes", [])
            keystrokes = data.get("json", {}).get("keystrokes", [])
            
            if not scenes:
                st.error("No scenes found in project data.")
                return None
                
            # Get zoom ranges and convert to click events
            zoom_ranges = scenes[0].get("zoomRanges", [])
            click_events = []
            for zr in zoom_ranges:
                x = zr["manualTargetPoint"]["x"]
                y = zr["manualTargetPoint"]["y"]
                start_time = zr["startTime"] / 1000.0
                click_events.append({
                    "time": start_time,
                    "x": x,
                    "y": y
                })
            
            # Get settings
            webcam_position = settings.get("webcam_position", "bottom-right")
            webcam_size = settings.get("webcam_size", 25)
            record_webcam = settings.get("record_webcam", False)
            cursor_settings = settings.get("cursor_settings", {})

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Could not open video file.")
            return None

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Handle timing
        if cut_end_sec is None or cut_end_sec <= 0:
            cut_end_sec = total_frames / fps
        total_duration = cut_end_sec - cut_start_sec

        # Setup output video
        out_path = os.path.join(project_folder, "zoomed_output.mp4")
        if vertical_mode:
            new_w = int(h * 9 / 16)
            if new_w > w:
                new_w = w
            out_size = (new_w, h)
        else:
            out_size = (w, h)

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(
            out_path, 
            fourcc, 
            fps, 
            out_size,
            True
        )

        if not out.isOpened():
            st.error("Failed to create output video file.")
            if 'cap' in locals():
                cap.release()
            return None

        # Frame processing loop
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
                
            current_sec = current_frame_idx / fps

            # Handle trimming
            if current_sec < cut_start_sec:
                current_frame_idx += 1
                continue
            if current_sec > cut_end_sec:
                break

            # Store original frame
            original_frame = frame_bgr.copy()

            # Apply effects in order:
            
            # 1. Background effects
            if background_settings and background_settings.get('background_type') != 'none':
                frame_bgr = customize_background(frame_bgr, background_settings)

            # 2. Apply zoom with configured intensity
            if zoom_ranges:
                active_zoom = get_active_zoom_range(zoom_ranges, current_sec)
                if active_zoom:
                    target_zoom = zoom_intensity
                    current_zoom = active_zoom.get("zoom", 1.0)
                    smoothed_zoom = current_zoom + (target_zoom - current_zoom) * zoom_smoothness
                    active_zoom["zoom"] = smoothed_zoom
                    
                    frame_bgr = compute_zoomed_frame(
                        frame_bgr, 
                        active_zoom,
                        original_frame if record_webcam else None,
                        webcam_position
                    )

            # 3. Motion blur
            if apply_motion_blur_flag:
                frame_bgr = apply_motion_blur(frame_bgr, blur_radius=blur_intensity)

            # 4. Keystrokes overlay
            if overlay_keys_flag:
                frame_bgr = overlay_keystrokes(
                    frame_bgr,
                    keystrokes,
                    current_sec,
                    style=key_style,
                    position=key_position
                )

            # 5. Mouse effects
            if show_mouse:
                current_clicks = [click for click in click_events 
                                if abs(click["time"] - current_sec) < 0.1]
                for click in current_clicks:
                    x = int(click["x"] * w)
                    y = int(click["y"] * h)
                    frame_bgr = apply_cursor_effects(
                        frame_bgr,
                        (x, y),
                        clicked=True,
                        style=click_style or "circle",
                        color=click_color or "#FF0000",
                        opacity=click_opacity
                    )

            # 6. Handle vertical mode
            if vertical_mode:
                x1 = (w - new_w) // 2
                x2 = x1 + new_w
                frame_bgr = frame_bgr[:, x1:x2]

            # Ensure frame size and format
            if frame_bgr.shape[:2][::-1] != out_size:
                frame_bgr = cv2.resize(frame_bgr, out_size)
            
            if len(frame_bgr.shape) == 2:
                frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
            elif frame_bgr.shape[2] == 4:
                frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGBA2BGR)

            # Handle speed factor
            if speed_factor != 1.0:
                if speed_factor > 1.0 and current_frame_idx % int(speed_factor) != 0:
                    current_frame_idx += 1
                    continue
                elif speed_factor < 1.0:
                    repeats = int(1 / speed_factor)
                    for _ in range(repeats):
                        out.write(frame_bgr)
                        frames_written += 1
                        if export_gif:
                            frames_for_gif.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                    current_frame_idx += 1
                    continue

            # Write frame
            out.write(frame_bgr)
            frames_written += 1
            if export_gif:
                frames_for_gif.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

            # Update progress
            current_frame_idx += 1
            if current_frame_idx % 30 == 0:
                progress = min((current_sec - cut_start_sec) / total_duration, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {current_frame_idx}/{total_frames}")

        # Cleanup
        progress_bar.progress(1.0)
        status_text.empty()
        cap.release()
        out.release()

        if frames_written == 0:
            st.error("No frames were written. Check your cut range and speed settings.")
            return None

        st.success("Video processing completed!")

        # Handle GIF export
        if export_gif and frames_for_gif:
            try:
                gif_path = os.path.join(project_folder, "zoomed_output.gif")
                clip = mpe.ImageSequenceClip(frames_for_gif, fps=fps)
                clip.write_gif(gif_path, fps=min(fps, 15))
                clip.close()
                
                with open(gif_path, "rb") as f:
                    st.download_button(
                        "Download GIF",
                        f,
                        file_name="screen_recording.gif",
                        mime="image/gif"
                    )
            except Exception as e:
                st.error(f"GIF export failed: {str(e)}")

        # Re-encode for compatibility
        final_mp4 = out_path
        reencoded_path = os.path.join(project_folder, "zoomed_output_qt.mp4")

        try:
            if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
                with st.spinner("Re-encoding video for compatibility..."):
                    success = reencode_with_libx264(out_path, reencoded_path)
                    if success and os.path.isfile(reencoded_path) and os.path.getsize(reencoded_path) > 0:
                        final_mp4 = reencoded_path
                        st.info("Successfully re-encoded for QuickTime compatibility.")
                    else:
                        st.warning("Re-encoding failed. Using original output.")
            else:
                st.error("Output video is empty or missing.")
                return None
        except Exception as e:
            st.error(f"Error during re-encoding: {str(e)}")
            final_mp4 = out_path

        # Show video preview
        if os.path.exists(final_mp4) and os.path.getsize(final_mp4) > 0:
            try:
                with open(final_mp4, 'rb') as video_file:
                    video_bytes = video_file.read()
                    if len(video_bytes) > 0:
                        st.video(video_bytes)
                        st.download_button(
                            label="Download Processed Video",
                            data=video_bytes,
                            file_name="processed_recording.mp4",
                            mime="video/mp4"
                        )
                    else:
                        st.error("Video file is empty")
            except Exception as e:
                st.error(f"Error previewing video: {str(e)}")
        else:
            st.error("No valid video file found for preview")

        return final_mp4

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        return None

# --------------------------------------------------------------------------
#  STREAMLIT UI
# --------------------------------------------------------------------------
def export_video(project_folder, format="mp4", quality="high", effects=None, 
                resolution=None, codec=None, include_audio=True, optimize_size=True, preset=None):
    """
    Export video with corrected FFmpeg filter syntax
    """
    import ffmpeg
    import os
    import json

    # Paths
    input_video = os.path.join(project_folder, "recording.mp4")
    audio_path = os.path.join(project_folder, "mic_audio.wav")
    project_json = os.path.join(project_folder, "project.json")
    output_path = os.path.join(project_folder, f"final_export.{format}")

    # Quality presets
    quality_presets = {
        'low': {
            'video_bitrate': '2000k',
            'audio_bitrate': '96k',
            'preset': 'ultrafast',
            'crf': 28
        },
        'medium': {
            'video_bitrate': '4000k',
            'audio_bitrate': '128k',
            'preset': 'medium',
            'crf': 23
        },
        'high': {
            'video_bitrate': '8000k',
            'audio_bitrate': '192k',
            'preset': 'slow',
            'crf': 18
        },
        'ultra': {
            'video_bitrate': '16000k',
            'audio_bitrate': '320k',
            'preset': 'veryslow',
            'crf': 16
        }
    }

    try:
        # Validate input video
        if not os.path.exists(input_video):
            raise FileNotFoundError(f"Input video not found: {input_video}")

        # Load project data
        with open(project_json, 'r', encoding='utf-8') as f:
            project_data = json.load(f)
            scenes = project_data.get("json", {}).get("scenes", [])
            zoom_ranges = scenes[0].get("zoomRanges", []) if scenes else []

        # Get quality settings
        preset_settings = quality_presets.get(quality.lower())
        if not preset_settings:
            raise ValueError(f"Unknown quality preset: {quality}")

        # Initialize stream
        stream = ffmpeg.input(input_video)
        
        # Prepare filter chains
        if effects and effects.get('zoom_enabled') and zoom_ranges:
            # Create individual segments for each zoom range
            segments = []
            for i, zr in enumerate(zoom_ranges):
                start_time = zr["startTime"] / 1000.0
                end_time = zr["endTime"] / 1000.0
                zoom_factor = zr["zoom"]
                x_factor = zr["manualTargetPoint"]["x"]
                y_factor = zr["manualTargetPoint"]["y"]
                
                # Split and process each segment
                segment = (
                    stream
                    .filter('trim', start=start_time, end=end_time)
                    .filter('setpts', 'PTS-STARTPTS')
                    .filter('scale', w=f'iw*{zoom_factor}', h=f'ih*{zoom_factor}')
                    .filter('crop', 
                           w=f'iw/{zoom_factor}', 
                           h=f'ih/{zoom_factor}',
                           x=f'iw*{x_factor}',
                           y=f'ih*{y_factor}')
                )
                segments.append(segment)
            
            # Concatenate all segments
            if segments:
                stream = ffmpeg.concat(*segments, v=1, a=0)
        
        # Apply motion blur if enabled
        if effects and effects.get('motion_blur'):
            blur_intensity = effects.get('blur_intensity', 2)
            stream = stream.filter('boxblur', blur_intensity)

        # Prepare output arguments
        output_args = {
            'vcodec': codec if codec else 'libx264',
            'preset': preset_settings['preset'],
            'crf': preset_settings['crf'],
            'video_bitrate': preset_settings['video_bitrate'],
            'pix_fmt': 'yuv420p',
            'movflags': '+faststart'
        }

        # Handle audio
        if include_audio and os.path.exists(audio_path):
            audio_stream = ffmpeg.input(audio_path)
            audio_stream = audio_stream.filter('aresample', 44100)
            
            # Merge audio and video
            stream = ffmpeg.concat(stream, audio_stream, v=1, a=1)
            
            output_args.update({
                'acodec': 'aac',
                'audio_bitrate': preset_settings['audio_bitrate'],
                'ar': '44100'
            })

        # Add output resolution if specified
        if resolution and resolution != "Original":
            if resolution == "4K":
                output_args.update({'s': '3840x2160'})
            elif resolution == "1080p":
                output_args.update({'s': '1920x1080'})
            elif resolution == "720p":
                output_args.update({'s': '1280x720'})

        # Create output
        stream = ffmpeg.output(stream, output_path, **output_args)

        # Run FFmpeg
        stream.overwrite_output().run(capture_stdout=True, capture_stderr=True)

        # Verify output
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            raise Exception("Export completed but output file is empty")

    except ffmpeg.Error as e:
        err_msg = e.stderr.decode() if e.stderr else str(e)
        raise Exception(f"FFmpeg error: {err_msg}")
    except Exception as e:
        raise Exception(f"Export error: {str(e)}")


def record_system_audio():
    """Record system audio using sounddevice"""
    import sounddevice as sd
    import soundfile as sf
    
    device_info = sd.query_devices(None, 'input')
    samplerate = int(device_info['default_samplerate'])
    channels = 2
    
    audio_data = sd.rec(
        int(duration_sec * samplerate),
        samplerate=samplerate,
        channels=channels
    )
    sd.wait()
    return audio_data, samplerate

def merge_audio_video(video_path, audio_path, output_path):
    """Merge audio and video using ffmpeg"""
    try:
        stream = (
            ffmpeg
            .input(video_path)
            .input(audio_path)
            .output(
                output_path,
                vcodec='copy',
                acodec='aac',
                strict='experimental'
            )
            .overwrite_output()
        )
        stream.run(capture_stdout=True, capture_stderr=True)
        return True
    except ffmpeg.Error as e:
        print(f"FFmpeg merge error: {e.stderr.decode()}")
        return False

def apply_cursor_effects(frame, cursor_pos, clicked=False, style="circle", color=(255, 255, 255), opacity=0.3):
    """
    Improved cursor effects with proper color handling
    """
    x, y = cursor_pos
    out = frame.copy()
    
    # Improved color conversion
    def parse_color(color):
        if isinstance(color, str) and color.startswith('#'):
            # Convert hex to RGB
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            return (b, g, r)  # OpenCV uses BGR
        return color
    
    color = parse_color(color)
    
    if style == "circle":
        # Simple circle highlight
        radius = 20
        cv2.circle(out, (x, y), radius, color, 2)
        if clicked:
            # Create expanding circles for click effect
            for r in range(radius, radius + 15, 5):
                alpha = max(0, 1 - (r - radius) / 15)
                overlay = out.copy()
                cv2.circle(overlay, (x, y), r, color, 2)
                cv2.addWeighted(overlay, alpha * opacity, out, 1 - alpha * opacity, 0, out)
                
    elif style == "ripple":
        # Ripple effect with multiple circles
        base_radius = 15
        for i in range(4):
            r = base_radius + i * 5
            alpha = max(0, 0.8 - i * 0.2)
            overlay = out.copy()
            cv2.circle(overlay, (x, y), r, color, 2)
            cv2.addWeighted(overlay, alpha * opacity, out, 1 - alpha * opacity, 0, out)
            
        if clicked:
            # Additional ripples for click
            for i in range(4):
                r = base_radius + 20 + i * 8
                alpha = max(0, 0.6 - i * 0.15)
                overlay = out.copy()
                cv2.circle(overlay, (x, y), r, color, 2)
                cv2.addWeighted(overlay, alpha * opacity, out, 1 - alpha * opacity, 0, out)
    
    elif style == "highlight":
        # Soft highlight with gradient
        radius = 25
        overlay = out.copy()
        
        # Create gradient effect
        for r in range(radius, 0, -5):
            alpha = (r / radius) * opacity
            cv2.circle(overlay, (x, y), r, color, -1)
            cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
            
        if clicked:
            # Add click highlight
            cv2.circle(out, (x, y), radius + 5, color, 2)
    
    return out

def smooth_cursor_movement(current_pos, target_pos, smoothing=0.5):
    """Smooth cursor movement using interpolation"""
    x = int(current_pos[0] + (target_pos[0] - current_pos[0]) * smoothing)
    y = int(current_pos[1] + (target_pos[1] - current_pos[1]) * smoothing)
    return (x, y)

def customize_background(frame, settings):
    """Apply comprehensive background customization effects"""
    if not settings or settings.get('background_type') == 'none':
        return frame

    out = frame.copy()
    
    # Apply main background effect
    if settings.get('background_type') == 'gradient':
        gradient = create_gradient(
            frame.shape,
            settings.get('gradient_colors', ['#000000', '#FFFFFF']),
            settings.get('gradient_direction', 'vertical')
        )
        opacity = settings.get('gradient_opacity', 0.3)
        cv2.addWeighted(out, 1 - opacity, gradient, opacity, 0, out)
        
    elif settings.get('background_type') == 'solid':
        color = settings.get('background_color', '#FFFFFF')
        if isinstance(color, str) and color.startswith('#'):
            color = hex_to_rgb(color)
        solid = np.ones_like(frame) * np.array(color, dtype=np.uint8)
        opacity = settings.get('background_opacity', 0.3)
        cv2.addWeighted(out, 1 - opacity, solid, opacity, 0, out)
        
    elif settings.get('background_type') == 'blur':
        blur_amount = settings.get('blur_amount', 20)
        blur_opacity = settings.get('blur_opacity', 1.0)
        blurred = cv2.GaussianBlur(out, (blur_amount*2+1, blur_amount*2+1), 0)
        cv2.addWeighted(out, 1 - blur_opacity, blurred, blur_opacity, 0, out)
        
    elif settings.get('background_type') == 'image' and settings.get('background_image'):
        bg_image = cv2.imdecode(
            np.frombuffer(settings['background_image'].read(), np.uint8),
            cv2.IMREAD_COLOR
        )
        bg_image = resize_background_image(
            bg_image,
            out.shape[:2],
            settings.get('image_fit', 'cover')
        )
        opacity = settings.get('background_opacity', 0.3)
        cv2.addWeighted(out, 1 - opacity, bg_image, opacity, 0, out)
    
    # Apply border if enabled
    if settings.get('border_enabled'):
        border_color = settings.get('border_color', '#FF0000')
        if isinstance(border_color, str) and border_color.startswith('#'):
            border_color = hex_to_rgb(border_color)
        border_width = settings.get('border_width', 5)
        border_opacity = settings.get('border_opacity', 1.0)
        
        if settings.get('border_style') == 'rounded':
            # Draw rounded border
            radius = border_width * 2
            overlay = out.copy()
            cv2.rectangle(
                overlay,
                (border_width, border_width),
                (out.shape[1] - border_width, out.shape[0] - border_width),
                border_color,
                border_width,
                cv2.LINE_AA
            )
            cv2.addWeighted(overlay, border_opacity, out, 1 - border_opacity, 0, out)
        else:
            # Draw regular border
            out = cv2.copyMakeBorder(
                out,
                border_width,
                border_width,
                border_width,
                border_width,
                cv2.BORDER_CONSTANT,
                value=border_color
            )
    
    # Apply padding if enabled
    if settings.get('padding_enabled'):
        padding = settings.get('padding_amount', 20)
        padding_color = settings.get('padding_color', '#FFFFFF')
        if isinstance(padding_color, str) and padding_color.startswith('#'):
            padding_color = hex_to_rgb(padding_color)
        
        padded = np.zeros((
            out.shape[0] + 2*padding,
            out.shape[1] + 2*padding,
            3
        ), dtype=np.uint8)
        padded[:] = padding_color
        
        # Insert original frame with padding
        padded[
            padding:-padding if padding > 0 else None,
            padding:-padding if padding > 0 else None
        ] = out
        
        # Apply padding opacity
        padding_opacity = settings.get('padding_opacity', 1.0)
        mask = np.zeros_like(padded, dtype=np.float32)
        mask[
            padding:-padding if padding > 0 else None,
            padding:-padding if padding > 0 else None
        ] = 1.0
        out = cv2.addWeighted(
            padded,
            padding_opacity,
            out,
            1 - padding_opacity,
            0
        )
    
    return out

def resize_background_image(image, target_size, fit_mode='cover'):
    """Resize background image according to fit mode"""
    th, tw = target_size
    ih, iw = image.shape[:2]
    
    if fit_mode == 'stretch':
        return cv2.resize(image, (tw, th))
    
    scale = max(tw/iw, th/ih) if fit_mode == 'cover' else min(tw/iw, th/ih)
    new_w, new_h = int(iw * scale), int(ih * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    
    # Center crop/pad to target size
    y1 = max(0, (new_h - th) // 2)
    x1 = max(0, (new_w - tw) // 2)
    y2 = min(new_h, y1 + th)
    x2 = min(new_w, x1 + tw)
    
    cropped = resized[y1:y2, x1:x2]
    
    # Pad if necessary
    if cropped.shape[:2] != target_size:
        result = np.zeros((th, tw, 3), dtype=np.uint8)
        y_offset = max(0, (th - cropped.shape[0]) // 2)
        x_offset = max(0, (tw - cropped.shape[1]) // 2)
        result[
            y_offset:y_offset + cropped.shape[0],
            x_offset:x_offset + cropped.shape[1]
        ] = cropped
        return result
    
    return cropped

def create_gradient(shape, colors, direction='vertical'):
    """
    Create a gradient background with multiple color stops
    """
    height, width = shape[:2]
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Convert hex colors to RGB
    rgb_colors = []
    for color in colors:
        if isinstance(color, str) and color.startswith('#'):
            rgb_colors.append(tuple(int(color[i:i+2], 16) for i in (1, 3, 5)))
        else:
            rgb_colors.append(color)
    
    if direction == 'vertical':
        for y in range(height):
            # Calculate position in gradient (0 to 1)
            pos = y / height
            
            # Find the two colors to interpolate between
            idx = pos * (len(rgb_colors) - 1)
            color1_idx = int(idx)
            color2_idx = min(color1_idx + 1, len(rgb_colors) - 1)
            
            # Calculate interpolation factor
            factor = idx - color1_idx
            
            # Interpolate between colors
            color1 = rgb_colors[color1_idx]
            color2 = rgb_colors[color2_idx]
            color = tuple(int(c1 + (c2-c1)*factor) for c1, c2 in zip(color1, color2))
            
            gradient[y, :] = color
            
    elif direction == 'horizontal':
        for x in range(width):
            pos = x / width
            idx = pos * (len(rgb_colors) - 1)
            color1_idx = int(idx)
            color2_idx = min(color1_idx + 1, len(rgb_colors) - 1)
            factor = idx - color1_idx
            
            color1 = rgb_colors[color1_idx]
            color2 = rgb_colors[color2_idx]
            color = tuple(int(c1 + (c2-c1)*factor) for c1, c2 in zip(color1, color2))
            
            gradient[:, x] = color
            
    elif direction == 'diagonal':
        for y in range(height):
            for x in range(width):
                pos = (x + y) / (width + height)
                idx = pos * (len(rgb_colors) - 1)
                color1_idx = int(idx)
                color2_idx = min(color1_idx + 1, len(rgb_colors) - 1)
                factor = idx - color1_idx
                
                color1 = rgb_colors[color1_idx]
                color2 = rgb_colors[color2_idx]
                color = tuple(int(c1 + (c2-c1)*factor) for c1, c2 in zip(color1, color2))
                
                gradient[y, x] = color
    
    return gradient

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_color):
    """Convert RGB tuple to hex color"""
    return '#{:02x}{:02x}{:02x}'.format(*rgb_color)

def get_export_preset(preset_name):
    """Get export settings for different presets"""
    presets = {
        'web': {
            'width': 1920,
            'height': 1080,
            'fps': 30,
            'bitrate': '2000k',
            'codec': 'libx264',
            'preset': 'medium'
        },
        'social': {
            'width': 1080,
            'height': 1920,
            'fps': 30,
            'bitrate': '2500k',
            'codec': 'libx264',
            'preset': 'medium'
        },
        'high_quality': {
            'width': 3840,
            'height': 2160,
            'fps': 60,
            'bitrate': '8000k',
            'codec': 'libx264',
            'preset': 'slow'
        }
    }
    return presets.get(preset_name, presets['web'])


def register_shortcuts():
    """Register global keyboard shortcuts"""
    shortcuts = {
        'start_recording': ['ctrl', 'shift', 'r'],
        'stop_recording': ['ctrl', 'shift', 's'],
        'pause_recording': ['ctrl', 'shift', 'p'],
        'toggle_webcam': ['ctrl', 'shift', 'w']
    }
    
    def on_activate():
        # Handle shortcut activation
        pass
        
    keyboard.add_hotkey(shortcuts['start_recording'], on_activate)

def main():
    st.title("🎥 Professional Screen Recorder")
    
    # Add sidebar for global settings
    with st.sidebar:
        st.header("Global Settings")
        theme = st.selectbox("Theme", ["Light", "Dark"])
        shortcuts_enabled = st.checkbox("Enable Keyboard Shortcuts", value=True)
        hardware_accel = st.checkbox("Hardware Acceleration", value=True)
        st.divider()
        
        if shortcuts_enabled:
            st.markdown("""
            ### Keyboard Shortcuts
            - **Start/Stop**: Ctrl+Shift+R
            - **Pause**: Ctrl+Shift+P
            - **Toggle Webcam**: Ctrl+Shift+W
            - **Toggle Audio**: Ctrl+Shift+A
            """)
    
    tabs = st.tabs(["Record", "Configure", "Edit", "Export"])
    
    with tabs[0]:  # Record tab
        st.subheader("Recording Setup")
        
        # Screen Selection
        screen_col, webcam_col, audio_col = st.columns(3)
        
        with screen_col:
            st.markdown("### Screen Capture")
            selected_screen = st.selectbox(
                "Select Screen",
                ["Main Screen", "Screen 1", "Screen 2", "Custom Region"]
            )
            
            if selected_screen == "Custom Region":
                st.number_input("Width", 100, 4096, 1920)
                st.number_input("Height", 100, 2160, 1080)
            
            fps = st.number_input("Frame Rate (FPS)", 1, 60, 30)
            duration = st.number_input("Duration (seconds)", 1, 3600, 30)
        
        with webcam_col:
            st.markdown("### Webcam Settings")
            record_webcam = st.checkbox("Include Webcam", value=True)
            
            if record_webcam:
                webcam_device = st.selectbox(
                    "Select Webcam",
                    ["Default Webcam", "Webcam 1", "Webcam 2"]
                )
                webcam_position = st.selectbox(
                    "Position",
                    ["top-left", "top-right", "bottom-left", "bottom-right"]
                )
                webcam_size = st.slider("Size (%)", 10, 40, 25)
                webcam_border = st.checkbox("Add Border", value=True)
                # In the webcam settings section
                if webcam_border:
                    border_color = st.color_picker("Border Color", "#FFFFFF", key="webcam_border_color")
        
        with audio_col:
            st.markdown("### Audio Settings")
            record_audio = st.checkbox("Record Audio", value=True)
            
            if record_audio:
                audio_source = st.multiselect(
                    "Audio Sources",
                    ["Microphone", "System Audio"],
                    default=["Microphone"]
                )
                if "Microphone" in audio_source:
                    mic_device = st.selectbox(
                        "Select Microphone",
                        ["Default Mic", "Mic 1", "Mic 2"]
                    )
                    mic_volume = st.slider("Mic Volume", 0, 100, 80)
                
                if "System Audio" in audio_source:
                    system_volume = st.slider("System Volume", 0, 100, 80)
        
        st.divider()
        
        # Advanced Settings
        quality_col, cursor_col = st.columns(2)
        
        with quality_col:
            st.markdown("### Quality Settings")
            quality_preset = st.select_slider(
                "Quality Preset",
                options=["Low", "Medium", "High", "Ultra"],
                value="High"
            )
            
            with st.expander("Advanced Quality Settings"):
                bitrate = st.select_slider(
                    "Bitrate",
                    options=["2 Mbps", "4 Mbps", "8 Mbps", "16 Mbps"],
                    value="8 Mbps"
                )
                encoding = st.selectbox(
                    "Encoding",
                    ["H.264", "H.265", "VP9"]
                )
        
        with cursor_col:
            st.markdown("### Cursor Effects")
            cursor_highlight = st.checkbox("Highlight Cursor", value=True)
            if cursor_highlight:
                # In the cursor_col section
                highlight_color = st.color_picker("Highlight Color", "#FFD700", key="highlight_color_record")
                click_color = st.color_picker("Click Color", "#FF0000", key="click_color_record")
                highlight_size = st.slider("Highlight Size", 10, 50, 20)
                highlight_opacity = st.slider("Highlight Opacity", 0, 100, 50)
            
            click_effects = st.checkbox("Click Effects", value=True)
            if click_effects:
                click_color = st.color_picker("Click Color", "#FF0000")
                click_animation = st.selectbox(
                    "Click Animation",
                    ["Ripple", "Circle", "None"]
                )
        
        # Recording Controls
        st.divider()
        col1, col2, col3 = st.columns([2,1,2])
        
        with col2:
            if st.button("Start Recording", use_container_width=True):
                with st.spinner("Recording in progress..."):
                    try:
                        # In the main function, update this section:
                        folder = record_screen_webcam(
                            duration_sec=duration,
                            fps=fps,
                            record_webcam=record_webcam,
                            webcam_position=webcam_position if record_webcam else "bottom-right",
                            webcam_size=webcam_size if record_webcam else 25,
                            record_audio=record_audio,
                            audio_sources=audio_source if record_audio else [],
                            quality=quality_preset.lower(),
                            cursor_highlight=cursor_highlight,
                            highlight_color=highlight_color if cursor_highlight else None,
                            highlight_size=highlight_size if cursor_highlight else 20,
                            # Remove this line causing the error:
                            # highlight_opacity=highlight_opacity/100 if cursor_highlight else 0.3,
                            click_effects=click_effects,
                            click_color=click_color if click_effects else None,
                            click_animation=click_animation if click_effects else None,
                            selected_screen=selected_screen,
                            encoding=encoding,
                            bitrate=bitrate
                        )
                        st.session_state["last_recording"] = folder
                        st.success("Recording completed!")
                    except Exception as e:
                        st.error(f"Recording failed: {str(e)}")
    
    with tabs[1]:  # Configure tab
        st.subheader("Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Background Settings")
            bg_type = st.selectbox(
                "Background Type",
                ["None", "Solid Color", "Gradient", "Image", "Blur"]
            )
            
            if bg_type == "Solid Color":
                bg_color = st.color_picker("Background Color", "#FFFFFF", key="bg_color_config")
            elif bg_type == "Gradient":
                gradient_direction = st.select_slider(
                    "Direction",
                    options=["Horizontal", "Vertical", "Diagonal"]
                )
                gradient_colors = st.multiselect(
                    "Colors",
                    ["#FF0000", "#00FF00", "#0000FF", "#FFFF00"],
                    ["#FF0000", "#0000FF"]
                )
            elif bg_type == "Image":
                bg_image = st.file_uploader("Upload Background Image")
            elif bg_type == "Blur":
                blur_amount = st.slider("Blur Amount", 0, 50, 20)
        
        with col2:
            st.markdown("### Export Presets")
            st.checkbox("Create YouTube Ready Videos", value=True)
            st.checkbox("Optimize for Social Media", value=True)
            st.checkbox("Enable High-Quality Mode", value=True)
            
            with st.expander("Default Export Settings"):
                st.selectbox(
                    "Format",
                    ["MP4", "GIF", "WebM"]
                )
                st.select_slider(
                    "Quality",
                    options=["Low", "Medium", "High", "Ultra"]
                )
    
    with tabs[2]:  # Edit tab
        if "last_recording" in st.session_state:
            st.subheader("Edit Recording")
            
            # Timeline control
            timeline_container = st.container()
            timeline_container.slider("Timeline", 0.0, 100.0, 0.0)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Enhancement")
                smart_zoom = st.checkbox("Smart Zoom", value=True)
                if smart_zoom:
                    zoom_intensity = st.slider("Zoom Intensity", 1.0, 3.0, 2.0)
                    zoom_smoothness = st.slider("Smoothness", 0.0, 1.0, 0.5)
                
                motion_blur = st.checkbox("Motion Blur", value=True)
                if motion_blur:
                    blur_intensity = st.slider("Blur Intensity", 1, 10, 2)

                # Background Settings
                st.markdown("### Background")
                bg_type = st.selectbox(
                    "Background Type",
                    ["None", "Solid Color", "Gradient", "Blur", "Image"]
                )
                
                background_settings = {
                    "background_type": bg_type.lower(),
                    "background_opacity": st.slider("Background Opacity", 0.0, 1.0, 0.3),
                }
                
                if bg_type == "Gradient":
                    background_settings.update({
                        "gradient_colors": st.multiselect(
                            "Gradient Colors",
                            ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"],
                            default=["#FF0000", "#0000FF"]
                        ),
                        "gradient_direction": st.select_slider(
                            "Direction",
                            options=["horizontal", "vertical", "diagonal"],
                            value="vertical"
                        ),
                        "gradient_opacity": st.slider("Gradient Opacity", 0.0, 1.0, 0.3)
                    })
                    
                elif bg_type == "Solid Color":
                    background_settings.update({
                        "background_color": st.color_picker("Background Color", "#FFFFFF", key="bg_color_edit"),
                        })
                    
                elif bg_type == "Blur":
                    background_settings.update({
                        "blur_amount": st.slider("Blur Amount", 0, 50, 20),
                        "blur_opacity": st.slider("Blur Opacity", 0.0, 1.0, 1.0)
                    })
                    
                elif bg_type == "Image":
                    background_image = st.file_uploader("Background Image", type=["jpg", "png"])
                    if background_image:
                        background_settings.update({
                            "background_image": background_image,
                            "image_fit": st.selectbox(
                                "Image Fit",
                                ["stretch", "contain", "cover"]
                            )
                        })
            
            with col2:
                st.markdown("### Overlays")
                show_keys = st.checkbox("Show Keystrokes", value=True)
                if show_keys:
                    key_position = st.selectbox(
                        "Position",
                        ["Bottom", "Top", "Left", "Right"]
                    )
                    key_style = st.selectbox(
                        "Style",
                        ["Modern", "Classic", "Minimal"]
                    )
                
                show_mouse = st.checkbox("Show Mouse Clicks", value=True)
                if show_mouse:
                    click_style = st.selectbox(
                        "Click Style",
                        ["Circle", "Ripple", "Highlight"]
                    )
                    click_color = st.color_picker("Click Color", "#FF0000", key="click_color_edit")
                    click_opacity = st.slider("Click Effect Opacity", 0.0, 1.0, 0.7)

                # Border Settings
                border_enabled = st.checkbox("Add Border", value=False)
                if border_enabled:
                    background_settings.update({
                        "border_enabled": True,
                        "border_color": st.color_picker("Border Color", "#FF0000", key="border_color_edit"),
                        "border_width": st.slider("Border Width", 1, 20, 5),
                        "border_opacity": st.slider("Border Opacity", 0.0, 1.0, 1.0),
                    })
            
            with col3:
                st.markdown("### Timing")
                speed = st.slider("Playback Speed", 0.25, 2.0, 1.0)
                trim_start = st.number_input("Trim Start (seconds)", 0.0)
                trim_end = st.number_input("Trim End (seconds)", 0.0)

                # Video Settings
                st.markdown("### Video Settings")
                vertical_mode = st.checkbox("Vertical Mode", value=False)
                if vertical_mode:
                    aspect_ratio = st.selectbox(
                        "Aspect Ratio",
                        ["9:16", "4:5", "1:1"]
                    )

                # Padding Settings
                padding_enabled = st.checkbox("Add Padding", value=False)
                if padding_enabled:
                    background_settings.update({
                        "padding_enabled": True,
                        "padding_amount": st.slider("Padding Amount", 0, 100, 20),
                        "padding_color": st.color_picker("Padding Color", "#FFFFFF", key="padding_color_edit"),
                        "padding_opacity": st.slider("Padding Opacity", 0.0, 1.0, 1.0)
                    })

            # Preview Controls
            st.divider()
            preview_col1, preview_col2 = st.columns(2)
            
            with preview_col1:
                generate_gif = st.checkbox("Generate GIF", value=False)
                if generate_gif:
                    gif_quality = st.select_slider(
                        "GIF Quality",
                        options=["Low", "Medium", "High"],
                        value="Medium"
                    )
            
            with preview_col2:
                preview_quality = st.select_slider(
                    "Preview Quality",
                    options=["Low", "Medium", "High"],
                    value="Medium"
                )

            if st.button("Preview Changes", use_container_width=True):
                with st.spinner("Processing video..."):
                    play_with_zoom(
                        st.session_state["last_recording"],
                        apply_motion_blur_flag=motion_blur,
                        blur_intensity=blur_intensity if motion_blur else 2,
                        overlay_keys_flag=show_keys,
                        key_style=key_style if show_keys else "Modern",
                        key_position=key_position if show_keys else "Bottom",
                        vertical_mode=vertical_mode,
                        speed_factor=speed,
                        cut_start_sec=trim_start,
                        cut_end_sec=trim_end if trim_end > 0 else None,
                        zoom_intensity=zoom_intensity if smart_zoom else 2.0,
                        zoom_smoothness=zoom_smoothness if smart_zoom else 0.5,
                        background_settings=background_settings,
                        show_mouse=show_mouse,
                        click_style=click_style if show_mouse else None,
                        click_color=click_color if show_mouse else None,
                        click_opacity=click_opacity if show_mouse else 0.7,
                        export_gif=generate_gif,
                        gif_quality=gif_quality if generate_gif else "Medium",
                        preview_quality=preview_quality
                    )
        else:
            st.info("Record a video first!")
    
    with tabs[3]:  # Export tab
        if "last_recording" in st.session_state:
            st.subheader("Export")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Format Settings")
                export_format = st.selectbox(
                    "Format",
                    ["MP4", "GIF", "WebM", "Vertical MP4"]
                )
                
                export_quality = st.select_slider(
                    "Quality",
                    options=["Low", "Medium", "High", "Ultra"],
                    value="High"
                )
                
                if export_format in ["MP4", "WebM", "Vertical MP4"]:
                    resolution = st.selectbox(
                        "Resolution",
                        ["Original", "4K", "1080p", "720p"]
                    )
                    
                    codec = st.selectbox(
                        "Codec",
                        ["H.264", "H.265", "VP9"]
                    )
            
            with col2:
                st.markdown("### Export Options")
                include_audio = st.checkbox("Include Audio", value=True)
                optimize_size = st.checkbox("Optimize File Size", value=True)
                generate_preview = st.checkbox("Generate Preview", value=True)
                
                export_preset = st.selectbox(
                    "Export Preset",
                    ["Custom", "YouTube", "Twitter", "Instagram"]
                )
            
            if st.button("Export", use_container_width=True):
                with st.spinner("Exporting..."):
                    try:
                        export_path = export_video(
                            st.session_state["last_recording"],
                            format=export_format.lower(),
                            quality=export_quality.lower(),
                            effects={
                                'zoom_enabled': smart_zoom,
                                'motion_blur': motion_blur,
                                'blur_intensity': blur_intensity
                            },
                            resolution=resolution,
                            codec=codec,
                            include_audio=include_audio,
                            optimize_size=optimize_size,
                            preset=export_preset
                        )
                        st.success("Export completed!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            with open(export_path, "rb") as f:
                                st.download_button(
                                    "Download Export",
                                    f,
                                    file_name=f"screen_recording.{export_format.lower()}"
                                )
                        
                        if generate_preview:
                            with col2:
                                st.video(export_path)
                                
                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")
        else:
            st.info("Record a video first!")

if __name__ == "__main__":
    main()