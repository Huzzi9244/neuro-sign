"""
Neuro-Sign | Phase 3: FastAPI WebSocket Server
================================================
Real-time gesture recognition backend using WebSockets.

Endpoints:
    GET  /           → Health check
    GET  /gestures   → List available gestures
    WS   /ws/predict → Real-time gesture prediction

Run:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import json
import logging
from collections import deque
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF info logs

import tensorflow as tf
from tensorflow import keras

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
_APP_DIR       = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR   = os.path.dirname(_APP_DIR)
MODEL_PATH     = os.path.join(_BACKEND_DIR, "ml_pipeline", "saved_models", "action_model.keras")
LABELS_PATH    = os.path.join(_BACKEND_DIR, "ml_pipeline", "saved_models", "gesture_labels.json")

SEQUENCE_LENGTH = 30      # frames needed for prediction
NUM_FEATURES    = 126     # 2 hands × 21 landmarks × (x, y, z)
CONFIDENCE_THRESHOLD = 0.6  # minimum confidence to report a gesture

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neuro-sign")

# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODEL AND LABELS
# ──────────────────────────────────────────────────────────────────────────────
def load_model_and_labels():
    """Load the trained Keras model and gesture label mapping."""
    
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}\n"
            "Please run the training notebook first."
        )
    
    if not os.path.isfile(LABELS_PATH):
        raise FileNotFoundError(
            f"Labels not found at: {LABELS_PATH}\n"
            "Please run the training notebook first."
        )
    
    logger.info(f"Loading model from: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH)
    
    logger.info(f"Loading labels from: {LABELS_PATH}")
    with open(LABELS_PATH, "r") as f:
        label_map = json.load(f)
    
    # Invert: {gesture_name: idx} → {idx: gesture_name}
    idx_to_label = {int(v): k for k, v in label_map.items()}
    
    logger.info(f"Loaded {len(idx_to_label)} gestures: {list(label_map.keys())}")
    
    return model, label_map, idx_to_label


# Load on startup
model, label_map, idx_to_label = load_model_and_labels()

# ──────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Neuro-Sign API",
    description="Real-time hand gesture recognition using LSTM",
    version="1.0.0",
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# REST ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "neuro-sign",
        "model_loaded": model is not None,
        "gestures_count": len(label_map),
    }


@app.get("/gestures")
async def list_gestures():
    """List all available gesture labels."""
    return {
        "gestures": list(label_map.keys()),
        "count": len(label_map),
    }


# ──────────────────────────────────────────────────────────────────────────────
# WEBSOCKET ENDPOINT
# ──────────────────────────────────────────────────────────────────────────────
class GesturePredictor:
    """Manages a rolling window of frames for a single WebSocket connection."""
    
    def __init__(self):
        self.frame_buffer: deque = deque(maxlen=SEQUENCE_LENGTH)
        self.last_prediction: Optional[str] = None
        self.last_confidence: float = 0.0
    
    def add_frame(self, landmarks: list) -> Optional[dict]:
        """
        Add a frame of landmarks and predict if we have enough frames.
        
        Args:
            landmarks: Flat list of 126 floats (2 hands × 21 landmarks × 3 coords)
        
        Returns:
            Prediction dict if ready, None otherwise
        """
        # Validate input
        if len(landmarks) != NUM_FEATURES:
            logger.warning(f"Invalid frame size: {len(landmarks)}, expected {NUM_FEATURES}")
            return None
        
        # Add to buffer
        self.frame_buffer.append(np.array(landmarks, dtype=np.float32))
        
        # Not enough frames yet
        if len(self.frame_buffer) < SEQUENCE_LENGTH:
            return {
                "status": "buffering",
                "frames_collected": len(self.frame_buffer),
                "frames_needed": SEQUENCE_LENGTH,
            }
        
        # Run prediction
        return self._predict()
    
    def _predict(self) -> dict:
        """Run model inference on the current frame buffer."""
        # Shape: (1, 30, 126)
        sequence = np.array(list(self.frame_buffer), dtype=np.float32)
        sequence = np.expand_dims(sequence, axis=0)
        
        # Predict
        predictions = model.predict(sequence, verbose=0)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx])
        
        gesture = idx_to_label.get(predicted_idx, "unknown")
        
        # Update state
        self.last_prediction = gesture
        self.last_confidence = confidence
        
        # Only report if confidence is above threshold
        if confidence >= CONFIDENCE_THRESHOLD:
            return {
                "status": "prediction",
                "gesture": gesture,
                "confidence": round(confidence, 4),
                "all_scores": {
                    idx_to_label[i]: round(float(predictions[0][i]), 4)
                    for i in range(len(predictions[0]))
                },
            }
        else:
            return {
                "status": "low_confidence",
                "gesture": gesture,
                "confidence": round(confidence, 4),
                "threshold": CONFIDENCE_THRESHOLD,
            }
    
    def clear(self):
        """Clear the frame buffer."""
        self.frame_buffer.clear()
        self.last_prediction = None
        self.last_confidence = 0.0


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """
    WebSocket endpoint for real-time gesture prediction.
    
    Expected JSON format from client:
    {
        "landmarks": [x1, y1, z1, x2, y2, z2, ..., x42, y42, z42]  // 126 floats
    }
    
    Or to clear the buffer:
    {
        "action": "clear"
    }
    """
    await websocket.accept()
    logger.info("WebSocket client connected")
    
    predictor = GesturePredictor()
    
    try:
        while True:
            # Receive JSON message
            data = await websocket.receive_json()
            
            # Handle control actions
            if "action" in data:
                if data["action"] == "clear":
                    predictor.clear()
                    await websocket.send_json({
                        "status": "cleared",
                        "message": "Frame buffer cleared",
                    })
                    continue
            
            # Handle landmark data
            if "landmarks" in data:
                landmarks = data["landmarks"]
                
                result = predictor.add_frame(landmarks)
                
                if result:
                    await websocket.send_json(result)
            else:
                await websocket.send_json({
                    "status": "error",
                    "message": "Missing 'landmarks' field in request",
                })
    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# STARTUP / SHUTDOWN
# ──────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 50)
    logger.info("  Neuro-Sign API Server Started")
    logger.info("=" * 50)
    logger.info(f"  Model: {MODEL_PATH}")
    logger.info(f"  Gestures: {list(label_map.keys())}")
    logger.info("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Neuro-Sign API Server shutting down...")
