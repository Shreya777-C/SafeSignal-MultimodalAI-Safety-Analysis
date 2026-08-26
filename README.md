# Multimodal Public Safety Analysis

This repository contains code from my final semester project focused on public safety analysis using audio, video, and crime incident data.

## Project Overview

The project combines multiple AI/data analysis pipelines:

1. **911/Police Call Audio Analysis**
   - Converts audio files to 16 kHz mono WAV format using FFmpeg.
   - Transcribes calls using Faster-Whisper.
   - Restores punctuation using a transformer model.
   - Detects text-based emotions from transcripts.
   - Extracts audio-based emotional signals such as valence, activation, dominance, and distress score.
   - Extracts prosody features such as pitch, intensity, voiced fraction, and speech rate.
   - Generates transcript outputs, SRT subtitle files, JSON outputs, and emotion metric plots.
   - Optionally uses Ollama for call context summarization.

2. **Video Emotion and Pose Analysis**
   - Uses YOLO pose estimation to analyze body movement from video.
   - Uses DeepFace for facial emotion detection.
   - Tracks video-level behavioral and emotional signals.

3. **Memphis Public Safety / Crime Data Analysis**
   - Analyzes MPD public safety incident data.
   - Combines crime data with Census ACS demographic variables.
   - Builds crime predictor datasets by tract and time period.
   - Performs correlation analysis, geospatial visualization, and predictive modeling.

## Repository Structure

```text
.
├── analysis_call.py              # Audio call analysis pipeline
├── Video_emo.ipynb               # Video pose and emotion analysis notebook
├── MPD_Analysis.ipynb            # Crime/public safety data analysis notebook
├── requirements.txt              # Python dependencies
├── .gitignore                    # Files/folders Git should ignore
└── README.md                     # Project documentation
```

## Main Files

### `analysis_call.py`
Runs the audio analysis pipeline. The script expects audio files inside a local folder such as:

```text
~/Downloads/call
```

It writes outputs to:

```text
~/Downloads/new_out
```

Generated outputs may include:

- `input_16k_mono.wav`
- `transcript.txt`
- `transcript.srt`
- `transcript.json`
- `emotion_metrics.png`
- `call_context.json`
- `call_context.txt`

### `Video_emo.ipynb`
Notebook for video-level emotion/body-pose analysis using:

- OpenCV
- YOLOv8 pose model
- DeepFace
- Pandas
- Matplotlib

### `MPD_Analysis.ipynb`
Notebook for analyzing public safety incident data and creating crime prediction features using:

- Pandas
- GeoPandas
- Folium
- Census ACS data
- XGBoost
- Scikit-learn

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows PowerShell**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

You also need FFmpeg installed and available from the command line:

```bash
ffmpeg -version
```

### 4. Run the audio analysis script

Place your audio files in:

```text
~/Downloads/call
```

Then run:

```bash
python analysis_call.py
```

## Notes

Large files such as raw videos, audio recordings, datasets, generated outputs, model weights, and cache folders should not be pushed to GitHub. They are ignored using `.gitignore`.

## Disclaimer

This project is for academic and research purposes only. It should not be used as a final decision-making tool for law enforcement, emergency response, or public safety without expert validation, bias testing, and human review.
