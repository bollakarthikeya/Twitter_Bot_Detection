# Twitter Bot Detection

## Overview

Machine-learning analysis for distinguishing **Twitter bot accounts from human accounts** using account metadata collected through Twitter's Search API with `tweepy`.

The notebook also separates tweet-level data for exploratory/NLP analysis, while the implemented classification pipeline focuses on twitter account attributes to identify bots.

## Dataset

After removing duplicate records:

- **869 total accounts**
- **705 human accounts**
- **164 bots**

To address class imbalance, all 164 bot accounts were retained and **164 human accounts were sampled**, producing a balanced dataset of **328 accounts**.

Account features include:

- Favorites count
- Followers count
- Friends count
- Listed count
- Statuses count
- Verified status

## Pipeline

1. Load Twitter data from JSON.
2. Separate account-level and tweet-level attributes.
3. Remove duplicate account records.
4. Balance inputs across bot and human classes by random sampling
5. Split data into **training (262)** and **test (66)** observations.
6. Standardize numerical features.
7. Examine features' structure using **PCA**.
8. Train and evaluate classification models.

PCA did not provide a strong dimensionality-reduction advantage; the largest principal component only explaining about **33.6%** of variance.

## Models & Results

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | **87.88%** |
| Random Forest | **92.42%** |

For Logistic Regression, cross-validation selected **C = 1e-5** from the tested regularization values.

## Key Finding

**Random Forest achieved the best result at 92.42% test accuracy**, demonstrating that bot and human accounts can be effectively separated using account metadata by employing machine-learning techniques.

## Technologies

- Python
- Pandas / NumPy
- scikit-learn
- NLTK
- Matplotlib / Seaborn
- Tweepy / Twitter Search API
