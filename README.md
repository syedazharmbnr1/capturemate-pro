# CaptureMate Pro

A professional-grade screen recording and editing suite with advanced capture features, real-time effects, and comprehensive editing capabilities.

## 🎥 Feature Demonstrations

### Basic Recording
Clean professional recording with no background effects.
![Basic Recording](demos/withoutbg.mp4)

### Custom Background Effects
Advanced background customization with various effects.
![Background Effects](demos/withbg.mp4)

### Motion Blur Effects
Professional motion blur implementation with adjustable intensity.
![Motion Blur](demos/blurred%20video.mp4)

### Smart Zoom
Intelligent zoom functionality with smooth transitions.
![Zoom Effects](demos/zoomed_output.gif)

## ✨ Core Features

### 🎥 Professional Recording Studio
- **Multi-Display Recording**
  - Record any screen or custom region
  - Support for multiple monitors
  - Custom region selection (up to 4K)
  - Hardware-accelerated capture

- **Webcam Integration**
  - Picture-in-picture recording
  - Adjustable size (10-40%)
  - Multiple position options:
    - Top-left, Top-right
    - Bottom-left, Bottom-right
  - Optional border with color customization

- **Advanced Audio Capture**
  - Microphone recording
  - System audio capture
  - Multiple source support
  - Individual volume control
  - High-quality audio sync

- **Performance Settings**
  - Up to 60 FPS recording
  - Adjustable duration
  - Hardware acceleration
  - Quality presets (Low to Ultra)
  - Bitrate control (2-16 Mbps)

### 🎯 Real-time Effects

- **Cursor Enhancement**
  - Customizable highlighting
  - Click visualization
  - Color personalization
  - Size adjustment
  - Opacity control

- **Click Effects**
  - Multiple styles:
    - Ripple animation
    - Circle effect
    - Simple highlight
  - Color customization
  - Adjustable opacity
  - Size control

- **Keystroke Visualization**
  - Live keystroke display
  - Multiple styles:
    - Modern
    - Classic
    - Minimal
  - Position customization
  - Clean overlay design

### 🎨 Advanced Editing

- **Background Customization**
  - Solid colors
  - Gradient effects
  - Custom images
  - Blur effects
  - Opacity control

- **Layout Options**
  - Vertical video mode
  - Aspect ratios:
    - 9:16 (Stories/Reels)
    - 4:5 (Instagram)
    - 1:1 (Square)
  - Custom padding
  - Border effects

- **Motion Effects**
  - Smart zoom with intensity control
  - Motion blur
  - Smooth transitions
  - Click animations

- **Timeline Controls**
  - Precise trimming
  - Speed adjustment (0.25x - 2.0x)
  - Preview quality options
  - Frame-accurate editing

### 📤 Export Studio

- **Format Support**
  - MP4 (H.264/H.265)
  - WebM (VP9)
  - Optimized GIF
  - Vertical video

- **Quality Options**
  - Multiple presets:
    - Low (2 Mbps)
    - Medium (4 Mbps)
    - High (8 Mbps)
    - Ultra (16 Mbps)
  - Resolution up to 4K
  - Codec selection

- **Platform Optimization**
  - YouTube-ready settings
  - Social media presets
  - Custom configurations
  - Size optimization

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Windows 10/11, macOS 10.15+, or Linux
- 4GB RAM minimum (8GB recommended)
- OpenGL 2.0 compatible graphics

### Installation

1. Clone the repository:
```bash
git clone https://github.com/syedazharmbnr1/capturemate-pro.git
cd capturemate-pro
```

2. Create virtual environment:
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

## 🎮 Usage Guide

### Recording
1. Select screen/region
2. Configure webcam (optional)
3. Choose audio sources
4. Set quality preferences
5. Click 'Start Recording'

### Editing
1. Use timeline for navigation
2. Apply background effects
3. Adjust motion and zoom
4. Configure overlays
5. Preview changes
6. Download processed video

### Exporting
1. Choose format (MP4/WebM/GIF)
2. Select quality preset
3. Configure resolution
4. Enable/disable audio
5. Apply optimizations
6. Export and download

## ⚠️ Known Issues

1. **Export Module**
   - FFmpeg encoder issue with H.264
   - Preview works perfectly
   - Export may require codec adjustment
   - See [Issue #2](https://github.com/syedazharmbnr1/capturemate-pro/issues/2)

2. **System Audio**
   - Windows: Extra permissions needed
   - macOS: Additional drivers required
   - Linux: PulseAudio limitation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please check the [issues page](https://github.com/syedazharmbnr1/capturemate-pro/issues) for current tasks and submit PRs with improvements.