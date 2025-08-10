# Keyword Spotting with Microcontrollers

## Overview
This project implements a complete TinyML pipeline for keyword spotting on the **Arduino Tiny Machine Learning Kit**.  
It covers data preprocessing, model size estimation, training, model conversion to TensorFlow Lite Micro, quantization (post-training & quantization-aware training), and pruning (structured & unstructured) for optimized deployment on resource-constrained devices.

The work was completed as part of **ECE 5545: Machine Learning Hardware and Systems**.

---

## Table of Contents
1. [Project Structure](#project-structure)  
2. [Requirements](#requirements)  
3. [Implementation Steps](#implementation-steps)  
   - Preprocessing  
   - Model Size Estimation  
   - Training & Analysis  
   - Model Conversion & Deployment  
   - Quantization  
   - Pruning  
4. [Results Summary](#results-summary)  
5. [Repository Structure](#repository-structure)  
6. [References](#references)  

---

## Project Structure
The project is divided into six main parts:

1. **Preprocessing**
   - Recorded audio samples and visualized them in the time and frequency domains.
   - Generated Spectrograms, Mel Spectrograms, and MFCCs using `torchaudio` and `librosa`.
   - Discussed differences between representations and why preprocessing improves ML performance.

2. **Model Size Estimation**
   - Estimated Flash and RAM usage for the TinyConv model.
   - Calculated FLOPs and benchmarked inference on CPU, GPU, and MCU.
   - Example: TinyConv uses ~6.6% of MCU flash and ~0.78% RAM at batch size 1.

3. **Training & Analysis**
   - Trained TinyConv on the Speech Commands dataset (4 keywords: yes, no, silence, unknown).
   - Achieved ~91% test accuracy with minimal overfitting.
   - Plotted training/validation accuracy curves.

4. **Model Conversion & Deployment**
   - Converted PyTorch model → ONNX → TensorFlow SavedModel → TFLite (INT8).
   - Deployed to Arduino Nano 33 BLE.
   - Profiled inference time and compared MCU vs CPU vs GPU:
     - MCU ~88 ms, CPU ~1.31 ms, GPU ~0.0416 ms.

5. **Quantization**
   - Implemented Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT).
   - Compared accuracy across bit-widths (2–8 bits) and included **minifloat quantization** experiments.
   - Found QAT more robust than PTQ, especially at low bit-widths.

6. **Pruning**
   - Implemented **Unstructured Pruning** (L1, L2, L∞ norms) and **Structured Pruning** (channel-level).
   - Measured accuracy vs. parameters, FLOPs, and runtime (CPU & MCU) at different pruning thresholds.
   - Fine-tuning improved accuracy retention after pruning.

---

## Requirements
- **Hardware**
  - Arduino Nano 33 BLE (TinyML Kit)
  - Microphone for audio recording

- **Software**
  - Python 3.8+
  - PyTorch, Torchaudio, Librosa
  - TensorFlow Lite & TFLite Micro
  - ONNX & onnx-tf
  - Arduino IDE with TFLite Micro libraries
  - Google Colab (Pro/Pro+ recommended)

---

## Implementation Steps

### 1. Preprocessing
- Normalize audio to reduce amplitude variance.
- Convert raw waveform → Spectrogram → Mel Spectrogram → MFCC.
- Visualization scripts in `notebooks/preprocessing.ipynb`.

### 2. Model Size Estimation
- Compute Flash/RAM usage based on parameter count and data type.
- Measure FLOPs and benchmark on CPU/GPU.
- Example:
  ```text
  Flash: 68.2 KB (~6.6% of MCU)
  RAM: 0.007856 MB (~0.78% of MCU)
  FLOPs: 0.676M
