# Neuro-Sign: Real-Time Hand Gesture Translator

A full-stack machine learning project that uses a custom Sequential LSTM network to classify dynamic hand gestures from live webcam feeds.

This project demonstrates how spatio-temporal neural networks learn from sequential frame data to perform accurate real-time translation.

## Project Overview

The model translates continuous hand movements into text. Unlike static image classifiers, this system evaluates rolling windows of movement.

Custom dataset recorded via webcam

21 3D hand landmarks extracted per frame

30 frames per prediction sequence

Classes Predicted:

* Perfect

    * ThumbsUp

    * Wave

    * Heart

    * Peace

    * (Dynamically extendable via data collection script)

## Key Objectives

    * Real-time spatial data extraction using MediaPipe in the browser

    * Handling time-series data for gesture recognition

    * Low-latency client-server communication via WebSockets

    * End-to-end data collection, training, and inference pipeline

## Model Architecture

The custom deep learning model includes:

    * Input Layer (Sequence of 30 frames, flattened coordinates)

    * LSTM Layers (spatio-temporal feature extraction)

    * Dropout Layers (to reduce overfitting)

    * Fully Connected Dense Layers

    * Softmax Output Layer

## Data Preprocessing

    * Landmark extraction via MediaPipe Tasks Vision

    * Flattening (x, y, z) coordinates into 1D arrays per frame

    * Aggregating arrays into structured sequential windows

    * One-hot encoding of categorical labels

## Tech Stack

     * Python

     * TensorFlow / Keras

     * FastAPI & WebSockets

     * React & Vite

     * MediaPipe

     * NumPy

## Local Setup
### 1. Start the Backend (FastAPI & AI Model)

Open a terminal and run the following commands to set up the Python environment and start the server:
```bash

cd backend
python -m venv .venv
.venv\Scripts\activate
pip install fastapi uvicorn tensorflow opencv-python mediapipe numpy scikit-learn
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

```
### 2. Start the Frontend (React Application)

Open a second new terminal and run the following commands to install dependencies and start the UI:
```bash

cd frontend
npm install
npm run dev
```
Once the Vite server starts, open your browser and navigate to the local link provided in the terminal (usually http://localhost:5173/).

## Author

Huzaifa Sikander