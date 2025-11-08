# Twitter Sentiment Analysis using Machine Learning

## Background (Business Context)
Twitter captures real-time public opinion at massive scale, but its language is noisy (slang, emojis, repeated characters, URLs, @mentions) and unstructured, making it hard to analyze systematically. This project uses the Sentiment140 dataset—1.6 million labeled tweets from [Kaggle]([https://www.kaggle.com/datasets/kazanova/sentiment140/data]). The NLP pipeline covers normalization, URL/emoji/mention handling, de-noising, tokenization, stopword removal, lemmatization, and TF-IDF feature construction to enable reliable, scalable sentiment analysis on social media text.

## Problem Statement
How can we accurately and efficiently classify the sentiment (positive/negative) of incoming tweets—despite informal language, abbreviations, repeated characters, emojis, URLs, and mentions—so stakeholders can monitor public opinion at scale and over time?

## Goals
- **Modeling**: Deliver a robust, end-to-end sentiment classifier for tweets using TF-IDF features and evaluate SVM / LR / NB, selecting the best model on validation metrics.
- **Quality**: Achieve at least ~80% accuracy with balanced performance across classes (report accuracy, confusion matrix, and F1).
- **Productization**: Produce a clean preprocessing + inference pipeline that scores new, unseen tweets reliably.
