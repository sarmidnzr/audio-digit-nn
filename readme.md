# Audio Digit Neural Network

Built a spoken-digit classifier from scratch in Python using NumPy and CuPy, achieving 90%+ accuracy with real-time microphone inference.

## Demo
https://youtu.be/mvJFuRyheho

## Overview
This project implements an end-to-end audio classification system that recognizes spoken digits (0–9) from live microphone input.

The system records user audio, processes it through a custom feature extraction pipeline, and performs inference using a fully connected neural network implemented without high-level deep learning frameworks.

## Key Features
- Neural network implemented from scratch using NumPy/CuPy (no PyTorch/TensorFlow)
- Real-time inference from live microphone input
- End-to-end audio preprocessing pipeline:
  - silence trimming
  - normalization
  - feature extraction
  - batching and caching
- GPU-accelerated computation using CuPy
- Achieves ~90% classification accuracy on spoken digit data

## Dataset
Trained on the Google Speech Commands dataset:  
https://www.kaggle.com/datasets/yashdogra/speech-commands

## Training
- Model trained on Rutgers iLab GPU machines
- Saved weights stored in `weights.npz`
- Includes evaluation and tuning for improved generalization

## How It Works
1. User records a spoken digit through the GUI
2. Audio is preprocessed and converted into features
3. Features are passed into the neural network
4. Model outputs the predicted digit

## Setup

```bash
git clone https://github.com/sarmidnzr/audio-digit-nn.git
cd audio-digit-nn
pip install .
python -m get_aud