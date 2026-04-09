## lane-keeping-assist
Lane Keeping Assist with U-Net Segmentation  Real-time lane detection and steering assistance system trained on TuSimple dataset.  Analyzes dashcam video, calculates lane offset, generates steering commands (LEFT/RIGHT/STRAIGHT),  and provides confidence metrics + warnings. 

Pipeline: Input video → U-Net mask → Offset calculation → Commands → Real-time dashboard with FPS, angle, confidence.
Processing stages: Raw input → Binary lane mask → Overlay with offset metrics and visualization.

## Processing Pipeline

<div align="center">

![Lane detection pipeline](analysis/example1.jpeg)

| Исходное изображение | Предсказанная маска | Наложение на кадр |
|---------------------|-------------------|------------------|
| **Raw Input** | **U-Net Binary Mask** | **Lane Overlay** |

</div>


## Analyzed video example

<div align="center">

![Lane Keeping Assist](analysis/analyzed_video-2.gif)

**Processing pipeline:** U-Net mask → Lane overlay → Real-time offset/angle/confidence metrics

</div>

## Key findings

•	U-Net achieves stable 27 FPS real-time inference on Apple M4 with MPS acceleration

•	Model confidence remains consistent across diverse lighting/weather conditions

•	Zero lane departure warnings during 26-second test sequence

•	Lateral offset calculation enables precise LEFT/RIGHT/STRAIGHT steering decisions

## Quick start 

1. Clone

git clone https://github.com/YOUR_USERNAME/lane-keeping-assist.git
cd lane-keeping-assist

2. Install

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

3. Analyze video

python analysis/video_analyzer.py test_video.mp4

4. Live webcam demo

python live_demo/lane_keeping.py
