# CaptureMate Pro

Professional-grade screen recording and editing suite designed for content creators, educators, and professionals. Create stunning screen captures with advanced features and real-time editing capabilities.

![CaptureMate Pro](assets/banner.png)

## Key Features

### Professional Recording Studio
- **Multi-Display Support**: Record any screen or custom region with precision
- **Webcam Integration**: Picture-in-picture recording with customizable positioning
- **High-Performance Capture**: Support for up to 60 FPS recording
- **Multi-Channel Audio**: Record from multiple audio sources simultaneously
  - System audio capture
  - Microphone input
  - Custom audio source selection

### Advanced Effects Suite
- **Dynamic Cursor Effects**
  - Customizable cursor highlighting
  - Click visualization
  - Motion tracking
- **Real-time Overlays**
  - Keystroke visualization
  - Customizable borders and padding
  - Professional transitions
- **Smart Zoom Technology**
  - Intelligent focus detection
  - Smooth zoom transitions
  - Multiple zoom styles

### Professional Editing Tools
- **Background Customization**
  - Solid colors
  - Gradient effects
  - Custom image backgrounds
  - Smart blur effects
- **Layout Options**
  - Vertical video mode
  - Custom aspect ratios
  - Responsive design templates
- **Timeline Editor**
  - Precise trimming
  - Speed adjustments
  - Multi-track editing

### Export Studio
- **Format Support**
  - MP4 with H.264/H.265
  - WebM with VP9
  - Optimized GIF export
- **Quality Presets**
  - YouTube optimized
  - Social media ready
  - Professional 4K support
- **Custom Export Options**
  - Bitrate control
  - Resolution selection
  - Format-specific optimization

## Getting Started

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

2. Set up virtual environment:
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

## Known Issues and Limitations

1. **Export Module**: Currently, while the preview functionality works flawlessly, the final export process may encounter occasional issues. Our team is actively working on resolving these limitations.

2. **System Audio Capture**: 
   - Windows: Requires additional system permissions
   - macOS: May require installation of additional audio drivers
   - Linux: Limited to PulseAudio systems

## Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- FFmpeg team for video processing capabilities
- OpenCV community for image processing features
- Streamlit team for the web interface framework

## Support

For support, feature requests, or bug reports, please open an issue in our GitHub repository.