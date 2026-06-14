# Radar Signal Processing Pipeline

## Overview

This document describes the signal processing pipeline for radar-based presence detection and respiration monitoring.

---

# 1. Radar Sensor Principle

Radar sensors transmit electromagnetic waves and receive reflected signals from targets in the environment.

Using the Time-of-Flight (ToF) principle, the radar estimates the distance to a target by measuring the travel time of the transmitted signal and its reflection.

By analyzing the reflected signal, radar can estimate distance, angle, motion, and even small physiological movements such as respiration.

---

# 2. Range Data and IQ Signals

After signal acquisition, the received radar signal is converted into complex IQ (In-phase and Quadrature) samples.

Each range bin contains a complex value:

```text
Range Bin 0 → I + jQ
Range Bin 1 → I + jQ
Range Bin 2 → I + jQ
...
```

Each IQ value represents the reflected radar response at a specific distance.

## Magnitude

```math
Magnitude = \sqrt{I^2 + Q^2}
```

Magnitude represents reflection strength.

## Phase

```math
Phase = atan2(Q, I)
```

Phase contains fine motion information and is used for respiration analysis.

---

# 3. Multi-Antenna Processing

For multi-antenna radar systems, angular estimation is performed using beamforming.

## Conventional Beamforming (CBF)

- Simple implementation
- Low computational complexity
- Lower angular resolution

## Capon Beamforming (MVDR)

- High angular resolution
- Better interference suppression
- Higher computational complexity

Output:

```text
Range-Angle Heatmap
```

```text
Range
  ↑
  │
  │      ● Target
  │
  └────────────→ Angle
```

---

# Presence Detection Pipeline

```text
Raw IQ Data
    ↓
Range FFT
    ↓
Beamforming
    ↓
Range-Angle Heatmap
    ↓
Clutter Removal
    ↓
Detection
    ↓
Clustering
    ↓
Presence Estimation
```

## Step 1. Radar Signal Acquisition

Acquire raw IQ samples from all receive antennas.

## Step 2. Range FFT

Convert ADC samples into range profiles.

```text
ADC Samples
    ↓
Range FFT
    ↓
Range Profile
```

## Step 3. Beamforming

Apply:

- Conventional Beamforming
- Capon Beamforming

Output:

```text
Range-Angle Heatmap
```

## Step 4. DSP Preprocessing

### Static Clutter Removal

Remove stationary reflections from:

- Walls
- Furniture
- Static objects

Methods:

- Mean subtraction
- Background subtraction
- Exponential Moving Average (EMA)

```math
Background[n] =
\alpha Current[n] +
(1-\alpha) Background[n-1]
```

```math
Signal_{dynamic}=Signal_{current}-Background
```

### Band-Pass Filtering

Remove unwanted frequency components.

Typical motion band:

```text
0.1 Hz – 5 Hz
```

## Step 5. Target Detection

Methods:

- CFAR
- Peak Detection

Output:

```text
Target Candidates
```

## Step 6. Clustering

Group neighboring detections into individual targets.

### DBSCAN

Advantages:

- No need to define the number of targets
- Robust to noise
- Widely used in radar applications

### K-Means

Advantages:

- Fast implementation

Disadvantages:

- Number of clusters must be predefined

Output:

```text
Person 1
Person 2
Person 3
```

## Step 7. Presence Estimation

Final output:

```text
Presence = True / False
Number of People = N
Location = (Range, Angle)
```

---

# Respiration Detection Pipeline

```text
IQ Data
    ↓
Target Selection
    ↓
Phase Extraction
    ↓
Phase Unwrapping
    ↓
Clutter Removal
    ↓
Band-Pass Filter
    ↓
Interpolation
    ↓
FFT
    ↓
Respiration Rate
```

## Step 1. Target Localization

Use the presence detection pipeline to determine:

```text
Range Bin
Angle Bin
```

for the target.

## Step 2. Phase Extraction

```math
Phase = atan2(Q, I)
```

## Step 3. Phase Unwrapping

Radar phase is wrapped within:

```text
-π ~ +π
```

Apply phase unwrapping:

```text
Wrapped Phase
     ↓
Phase Unwrapping
     ↓
Continuous Phase
```

## Step 4. Clutter Removal

Remove:

- Static reflections
- Environmental drift
- Slow-varying background noise

Methods:

- Mean subtraction
- High-pass filtering
- EMA-based clutter suppression

## Step 5. Band-Pass Filtering

Respiration frequency range:

```text
0.1 Hz – 0.5 Hz
```

Equivalent to:

```text
6 – 30 breaths/min
```

## Step 6. Interpolation

Increase temporal resolution.

Methods:

- Linear interpolation
- Cubic spline interpolation

Purpose:

- Improve FFT resolution
- Improve peak estimation accuracy

## Step 7. FFT Analysis

```text
Filtered Signal
      ↓
FFT
      ↓
Frequency Spectrum
```

Extract:

```text
Peak Frequency
```

## Step 8. Respiration Rate Estimation

```math
Respiration\ Rate = Peak\ Frequency \times 60
```

Example:

```text
0.25 Hz × 60 = 15 BPM
```

---

# Summary

## Presence Detection

```text
IQ Data
→ Range FFT
→ Beamforming
→ Clutter Removal
→ Detection
→ Clustering
→ Presence Estimation
```

## Respiration Detection

```text
IQ Data
→ Target Selection
→ Phase Extraction
→ Phase Unwrapping
→ Clutter Removal
→ Band-Pass Filter
→ Interpolation
→ FFT
→ Respiration Rate Estimation
```
