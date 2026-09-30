# Fingernail Anemia Screening — Mobile Web Frontend

This directory contains the user interface for mobile camera capture and file upload screening.

## Features

- **Live Camera Reticle**: Real-time viewfinder with alignment guide box and tap-to-focus guidance.
- **Image File Upload**: Drag-and-drop or file picker supporting JPEG, PNG, and WebP formats.
- **Dual-Image Preview**: Simultaneous display of original input photograph and extracted 224×224 subungual ROI.
- **Intermediate Pipeline Trace**: Live transparent display of EfficientNet-B0 logit, JetX-GT probability, logistic fusion, and isotonic calibration.
- **Recent Session History**: In-browser session log of recent assessments with latencies and timestamps.
- **Medical Research Disclaimers**: Prominent non-diagnostic research labeling across all views.

## File Organization

```
frontend/
├── index.html       <- Semantic HTML5 user interface
├── app.js           <- Orchestration logic, camera management, API client
├── style.css        <- Theme, responsive CSS, glassmorphism, animations
├── assets/          <- Static icons, illustrations, and logos
└── README.md        <- Frontend documentation
```

## Running the Frontend

The frontend is served directly by the FastAPI backend at `http://localhost:8000/`.
Static assets are mounted at `/static`.
