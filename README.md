# Next Word Prediction using RNNs

A simple LSTM + GRU model to predict the next word in a sentence using sequential data.

## Overview

This project implements a next word prediction model using a combination of LSTM and GRU layers. The model is trained on sample text datasets to explore how language models assist with next-word suggestions.

## Project Structure

```
next_word_prediction/
├── data/
│   └── sample_text.txt
├── models/
├── model.py
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train the model:
   ```
   python train.py
   ```

3. Use the model for prediction:
   ```
   python predict.py
   ```

## Model Architecture

The model uses:
- Embedding layer to convert words to vectors
- LSTM layer for sequence processing
- GRU layer for additional sequence processing
- Dense layer with softmax activation for word prediction

## Usage

After training, you can use the model to predict the next words in a sentence by running the prediction script and entering your seed text.

## Future Improvements

- Train on larger datasets
- Experiment with different model architectures
- Add temperature parameter for controlling prediction randomness
- Implement beam search for better predictions