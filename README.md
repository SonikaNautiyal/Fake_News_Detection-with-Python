# Fake_News_Detection-with-Python


## Overview
This project demonstrates how to build a machine learning model to detect fake news using Python. It leverages the **TfidfVectorizer** for feature extraction and the **PassiveAggressiveClassifier** for classification. The goal is to classify news articles as either **REAL** or **FAKE** with high accuracy.

---

## Table of Contents
1. [Introduction](#introduction)  
2. [Technologies Used](#technologies-used)  
3. [Dataset](#dataset)  
4. [Implementation Steps](#implementation-steps)  
5. [Results](#results)  
6. [Usage Instructions](#usage-instructions)  
7. [Conclusion](#conclusion)  
8. [References](#references)  

---

## Introduction
Fake news is a type of yellow journalism that spreads misinformation, often to serve political or social agendas. It can include exaggerated claims and false narratives, which are amplified by social media algorithms and filter bubbles.

This project aims to create a machine learning model that identifies fake news articles and distinguishes them from real ones. It uses **scikit-learn** to implement text processing and machine learning algorithms.

---

## Technologies Used
- **Python 3.x**
- **scikit-learn** - For machine learning models and vectorization
- **pandas** - For data manipulation
- **numpy** - For numerical computations
- **matplotlib** and **seaborn** - For data visualization
- **Jupyter Notebook** - For development and testing

---

## Dataset
We use the **news.csv** dataset, which contains labeled news articles classified as **REAL** or **FAKE**. It has the following columns:
- **title** - Title of the news article
- **text** - Full text of the article
- **label** - Classification label (REAL or FAKE)

---
## Results
- **Accuracy**: Achieved an accuracy of approximately **93%** on the test dataset.
- **Confusion Matrix**: Demonstrates the performance of classification with true positives, true negatives, false positives, and false negatives.

---

## Usage Instructions
1. **Clone the Repository:**
```bash
git clone https://github.com/username/fake-news-detection.git
cd fake-news-detection
```
2. **Install Dependencies:**
```bash
pip install -r requirements.txt
```
3. **Run the Script:**
```bash
python fake_news_detector.py
```
4. **Dataset:** Place the dataset file (news.csv) in the project directory before running the script.

---

## Conclusion
This project demonstrates an effective method to detect fake news articles using machine learning. By leveraging the TF-IDF vectorizer and Passive Aggressive Classifier, it achieves high accuracy and is suitable for real-world applications.

---

## References
1. [Scikit-learn Documentation](https://scikit-learn.org/stable/index.html)  
2. [Dataset Source](https://www.kaggle.com/)  
3. TutorialsPoint, Machine Learning Techniques

