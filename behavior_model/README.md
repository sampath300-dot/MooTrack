# MooTrack - Behaviour Model

## Purpose
This module performs cattle behaviour recognition from images using Deep Learning.

## Main Model
ResNet18 or MobileNet using transfer learning.

## Input
A cattle image.

## Possible Behaviour Classes
The final classes will depend on the selected dataset. Possible classes include:
- Standing
- Lying
- Eating/Foraging
- Drinking
- Rumination

## Processing
Cattle image → preprocessing → CNN → classification

## Output
The model returns:
- Predicted behaviour
- Confidence score

## Main Functions
- Image preprocessing
- Model training/fine-tuning
- Model evaluation
- Behaviour prediction

## Prediction Function

```python
predict_behavior(image_path)
