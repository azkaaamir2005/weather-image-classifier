# weather-image-classifier
Edge AI weather hazard engine using MobileNetV3-Small (PyTorch) to classify CCTV and dashcam frames, bypassing slow hardware sensors. Features 5-class dataset scoping, saved confusion matrix profiling, a dedicated Tkinter/Matplotlib prediction GUI, and live public traffic camera feed ingestion with sub-20ms CPU latency

## Problem Statement & Architecture Strategy
Traditional weather alert networks rely on delayed physical hardware sensors. This project implements edge computer vision to detect hazardous road weather directly from camera feeds in real time. 

To eliminate visual ambiguity, the 11-class raw Kaggle weather dataset is programmatically filtered down to 5 visually distinct classes (`fogsmog`, `lightning`, `rain`, `sandstorm`, `snow`).

## Key Features
- **Engine:** MobileNetV3-Small transfer learning with a lightweight **~9 MB model footprint**.
- **Edge Performance:** Sub-20ms inference latency on integrated CPU graphics with ~91% test accuracy.
- **Evaluation & Profiling:** Automated background export of confusion matrix plots (`outputs/confusion_matrix.png`) for low-contrast boundary profiling (e.g., fog vs. rain).
- **Desktop Interface:** Standalone dual-panel GUI (`predict.py`) displaying live single-image predictions, confidence bar graphs, and inference timing.
- **Stretch Goal Integration:** Live public traffic camera snapshot ingestion via direct URL fetching (`predict_traffic_url`).

## Project Structure
```text
weather image classifier/
├── data/
│   └── weather_dataset/       # Raw multi-class dataset
├── outputs/
│   └── confusion_matrix.png   # Auto-generated evaluation chart
├── main.py                    # Training, dataset loader, matrix generator & CCTV stretch goal
├── predict.py                 # Standalone Tkinter/Matplotlib GUI application
└── weather_mobilenet.pth      # Saved model weights
