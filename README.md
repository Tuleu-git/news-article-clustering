# News Article Clustering

Exploring the semantic structure of news article titles using PCA, K-Means clustering, and Logistic Regression.

## Overview

This project investigates whether the semantic content of news article titles alone is sufficient to distinguish between topic categories, without reading the articles themselves.

The dataset is a 20,000-article sample from [AG News](https://huggingface.co/datasets/fancyzhx/ag_news), covering four categories: **World**, **Sports**, **Business**, and **Sci/Tech**. Each title is represented as a 384-dimensional sentence embedding produced by `all-MiniLM-L6-v2`, which maps semantically similar texts to nearby points in vector space. The central question is: does the geometry of this embedding space reflect the topical structure of the news?

## What's Inside

### Dimensionality Reduction (PCA)
The 384-dimensional embeddings are projected down to 3 principal components and visualised interactively in 3D. Even at just ~9% retained variance, business articles form a visually distinct region — suggesting their titles carry a particularly consistent vocabulary compared to other categories.

### Clustering (K-Means)
K-Means is applied to the 3 principal components. The elbow method points to k=4, which turns out to map almost perfectly to the four news categories — each cluster is dominated by one category. The most interesting boundary is between business and sci/tech, where shared vocabulary around companies and products creates genuine semantic overlap.

### Classification (PCA + Logistic Regression Pipeline)
A scikit-learn pipeline combines PCA with Logistic Regression to classify whether an article belongs to the "world" category. Hyperparameter tuning via 5-fold cross-validation selects the optimal number of PCA components (50), regularisation strength, and class weighting. The final model reaches **92% accuracy** and an **F1-score of 0.82** on the held-out test set — using only the article title.

## Results

| | |
|---|---|
| PCA (3 components) | Business cluster visually separates; other categories overlap |
| K-Means (k=4) | Near-perfect one-to-one mapping to the four news categories |
| Classification | 92% accuracy · F1 = 0.82 on world category |

## Setup

```bash
git clone https://github.com/Tuleu-git/news-article-clustering.git
cd news-article-clustering
pip install -r requirements.txt
jupyter notebook notebooks/news_clustering_analysis.ipynb
```

## Tech Stack

`scikit-learn` · `Plotly` · `Seaborn` · `Pandas` · `NumPy`
