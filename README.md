# Music Genre Classification (Audio Analysis)

**COMP-6915: Machine Learning — Winter 2026**
**Group 07:** Joshua Oseimobor (202492785), Covenant Akisanmi (202582097), Cletus Chiemerie (202492457)

## Overview

A comparative study of traditional machine learning (SVM, Random Forest) and deep learning (CNN) approaches for classifying audio signals into 10 musical genres using the GTZAN dataset.

## Setup

```bash
pip install -r requirements.txt
```

Place `archive.zip` (GTZAN dataset) in the project root directory.

## Usage

### Run the full pipeline
```bash
python run_all.py
```

### Run individual phases
```bash
python run_all.py --phase 1   # Data pipeline (extract, validate, split)
python run_all.py --phase 2   # Feature engineering + mel-spectrogram generation
python run_all.py --phase 3   # Model training (SVM, RF, CNN)
python run_all.py --phase 4   # Evaluation, feature importance, noise robustness
```

### Launch the web demo
```bash
cd webapp
python app.py
```
Then open `http://localhost:5000` in your browser.

## Project Structure

```
config.py                 # Central configuration
run_all.py                # Master pipeline orchestrator
src/
  data_pipeline.py        # Phase 1: data extraction, validation, splitting
  feature_engineering.py   # Phase 2: tabular feature preparation
  mel_spectrogram.py       # Phase 2: mel-spectrogram generation
  traditional_ml.py        # Phase 3a: SVM + Random Forest
  cnn_model.py             # Phase 3b: CNN architecture
  train_cnn.py             # Phase 3b: CNN training
  evaluate.py              # Phase 4: evaluation metrics
  feature_importance.py    # Phase 4: feature importance analysis
  noise_robustness.py      # Phase 4: noise experiment
notebooks/
  01_eda.ipynb             # Exploratory data analysis
webapp/                    # Flask web application
report/                    # IEEE LaTeX report
```

## Dataset

GTZAN: 1,000 audio tracks (10 genres x 100 tracks), 30 seconds each, 22,050 Hz WAV.

## Models

- **SVM** (RBF/linear kernel) on hand-crafted audio features
- **Random Forest** on hand-crafted audio features
- **CNN** (4-block architecture) on log-mel spectrograms
