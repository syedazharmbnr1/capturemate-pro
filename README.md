# CaptureMate Pro

A professional-grade screen recording and editing suite with advanced capture features, real-time effects, and comprehensive editing capabilities.

## Feature Demonstrations

### Base Recording
![Basic Recording](demos/withoutbg.mp4)
Clean screen capture with professional output and smooth cursor tracking.

### Background Effects
![Background Effects](demos/withbg.mp4)
Customizable background effects with gradients and color options.

### Motion Blur Effect
![Motion Blur](demos/blurred\ video.mp4)
Professional motion blur implementation with adjustable intensity.

### Smart Zoom
![Zoom Effect](demos/zoomed_output.gif)
Intelligent zoom capabilities with smooth transitions.

## Key Features

### Recording Studio
- **Multi-Screen Support**
  - Record any display or custom region
  - Support for multiple monitors
  - Custom region selection with precise dimensions
- **Webcam Integration**
  - Picture-in-picture recording
  - Adjustable webcam size (10-40%)
  - Multiple position options (top-left, top-right, bottom-left, bottom-right)
  - Optional webcam border with custom color
- **Audio Capture**
  - Microphone recording
  - System audio capture
  - Multiple audio source selection
  - Volume control for each source
- **Performance Options**
  - Frame rates up to 60 FPS
  - Customizable duration
  - Hardware acceleration support
  - Multiple quality presets

### Real-time Effects
- **Cursor Enhancement**
  - Customizable cursor highlighting
  - Click visualization effects
  - Multiple cursor styles
  - Color customization for cursor and clicks
- **Click Effects**
  - Ripple animation
  - Circle effect
  - Highlight style
  - Adjustable opacity and size
- **Keystroke Visualization**
  - Real-time keystroke display
  - Multiple style options (Modern, Classic, Minimal)
  - Customizable position
  - Clean overlay design

### Advanced Editing
- **Background Customization**
  - Solid color backgrounds
  - Gradient effects with multiple colors
  - Image backgrounds with fit options
  - Smart blur effects
- **Layout Options**
  - Vertical video mode (9:16, 4:5, 1:1)
  - Custom aspect ratios
  - Padding and border effects
  - Position adjustments
- **Motion Effects**
  - Smart zoom with intensity control
  - Motion blur with adjustable strength
  - Smooth transitions
  - Click effect animations
- **Timeline Editing**
  - Precise trimming controls
  - Playback speed adjustment (0.25x - 2.0x)
  - Preview quality options
  - Frame-by-frame control

### Export Options
- **Format Support**
  - MP4 (H.264/H.265)
  - WebM (VP9)
  - Optimized GIF export
  - Vertical video formats
- **Quality Settings**
  - Multiple quality presets (Low to Ultra)
  - Bitrate control (2-16 Mbps)
  - Resolution options up to 4K
  - Format-specific optimization
- **Export Presets**
  - YouTube-ready export
  - Social media optimization
  - Custom preset configuration
  - Size optimization options

## Installation

1. Clone the repository:
```bash
git clone https://github.com/syedazharmbnr1/capturemate-pro.git
cd capturemate-pro
```

2. Set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Launch CaptureMate Pro:
```bash
python app.py
```

## Usage Guide

### Recording
1. Select your screen or region
2. Configure webcam settings if needed
3. Choose audio sources
4. Set quality and effects preferences
5. Click "Start Recording"

### Editing
1. Use the timeline slider to navigate
2. Apply background effects and customization
3. Adjust motion effects and zoom
4. Configure overlays and visual elements
5. Preview changes in real-time
6. Download processed video in your preferred format

### Exporting
1. Choose export format (MP4/WebM/GIF)
2. Select quality preset
3. Configure resolution and codec
4. Enable/disable audio
5. Choose optimization options
6. Export and download the final video

## Known Issues

1. **Export Module**: Currently experiencing FFmpeg encoder issues. The preview functionality works perfectly, but there might be issues with the final export:
   - Error with H.264 encoder naming
   - Temporary workaround available
   - See [Issue #2](https://github.com/syedazharmbnr1/capturemate-pro/issues/2)

2. **System Audio Capture**: 
   - Windows: Requires additional permissions
   - macOS: Additional audio drivers might be needed
   - Linux: Limited to PulseAudio systems

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.