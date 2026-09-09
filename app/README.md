# MooTrack - Web Application

## Purpose
This module provides the web interface for the MooTrack cattle monitoring system.

## Technology
Streamlit

## Main Pages

### Dashboard
Displays:
- Total analyses
- Recent predictions
- Audio analysis results
- Behaviour analysis results

### Acoustic Analysis
Allows the user to:
- Upload cattle audio
- Run acoustic analysis
- View prediction
- View confidence score

### Behaviour Analysis
Allows the user to:
- Upload cattle image
- Run behaviour analysis
- View prediction
- View confidence score

### History
Displays previous analysis results.

## Model Integration

The application will later call:

```python
predict_audio(audio_path)
