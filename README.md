# Disaster Tweet Classification

Final project for MIT 15.776: Hands-On Deep Learning. Classifies tweets as real disaster announcements or not using a range of NLP models, from bag-of-words to fine-tuned transformers.

**Team:** Giuseppe Iannone, Luca Sfragara, Trisha Sutivong, Hanna Zhang

## Overview

Dataset from the Kaggle NLP with Disaster Tweets competition: 7,613 labeled tweets. Each tweet is labeled 1 (real disaster) or 0 (not a disaster). The dataset includes tweet text, an optional keyword, and an optional location field.

We split the training data into train (68%), validation (17%), and test (15%) sets with stratification, since the original Kaggle test set is unlabeled.

## Models

| Model | Notes |
|---|---|
| Baseline | Majority class prediction |
| Bag of Words | Multi-hot vectorization, dense classifier |
| BoW (Tuned) | Keras Tuner hyperparameter search |
| BoW + Features | Added keyword and location features |
| Transformer Encoding | Custom transformer architecture |
| Transformer + Features | With engineered features |
| RoBERTa | Fine-tuned with PyTorch, early stopping on F1 |
| DistilBERT | Fine-tuned with PyTorch |
| Ensemble | Soft voting: BoW (20%) + DistilBERT (40%) + RoBERTa (40%) |

Primary metric is F1 score, following the Kaggle competition guidelines.

## Stack

Python, TensorFlow/Keras, PyTorch, HuggingFace Transformers, Keras Tuner, scikit-learn
