# **Midtermn Project ML Zoomcamp 2024 - Bankruptcy Prediction Service**

This service predicts the probability of company bankruptcy based on financial metrics.

## Problem Description

This project focuses on predicting bankruptcy risk for American public companies listed on the NYSE and NASDAQ. The goal is to develop a machine learning model that can effectively identify companies at risk of bankruptcy based on their financial indicators.

### Dataset Overview
- **Time Period**: 1999-2018
- **Companies**: 8,262 distinct companies
- **Total Observations**: 78,682 firm-year combinations
- **Data Quality**: Complete dataset with no missing values, synthetic entries, or imputed values

### Definition of Bankruptcy
According to the SEC, a company is considered bankrupt under two conditions:
1. Filing Chapter 11 (Reorganization)
  - Company continues operations
  - Management maintains control
  - Major business decisions require bankruptcy court approval
2. Filing Chapter 7 (Liquidation)
  - Complete cessation of operations
  - Company goes out of business

### Target Variable
- **Label 1 (Bankruptcy)**: Company filed for Chapter 11 or Chapter 7 in the following year
- **Label 0 (No Bankruptcy)**: Company continues normal operations

### Data Split
The dataset is chronologically divided into:
- **Training Set**: 1999-2011
- **Validation Set**: 2012-2014
- **Test Set**: 2015-2018

This temporal split ensures that the model's evaluation reflects its real-world performance on future, unseen cases, making it a practical tool for bankruptcy risk assessment.

## Project Structure

```text
midtermn_project/
├── data/
│   └── american_bankruptcy.csv
├── models/
│   └── model_C=1.0.bin
├── noteboks/
│   └── notebook.ipynb
├── scripts/
│   ├── predict.py
│   └── test_predictions.py
│   └── train.py
├── Dockerfile
└── requirements.txt
```
## Prerequisites

- Docker installed on your machine
- Python 3.9 or later (if you want to run test script locally)

### 1. Clone the Repository
```git
git clone [https://github.com/AlexanderPelaezJimenez/midtermn_project_mlzoomcamp_2024.git]

```
```bash
cd midtermn_project_mlzoomcamp_2024
```
### 2. Build and Run the Docker Container
# Build the Docker image
```docker
docker build -t bankruptcy-prediction .
```

# Run the container
```docker
docker run -it --rm -p 9696:9696 bankruptcy-prediction
```

### 3. Make Predictions
First, install the required package:
```python
pip install requests
```
Then run the test script:
```python
python scripts/test_predictions.py
```

**API Response Format**
The service returns a JSON with the following structure:
```json
{
    "probability_no_bankruptcy": 0.9628759765625,
    "probability_bankruptcy": 0.037124024437499995,
    "prediction": "No bankruptcy risk"
}
```
Where:
```text
probability_no_bankruptcy: Probability that the company will not go bankrupt
probability_bankruptcy: Probability that the company will go bankrupt
prediction: Final prediction based on a 0.5 threshold
```

## Input Features

The model requires the following financial indicators:

| Feature | Description |
|---------|-------------|
| current_ratio | Current Assets / Current Liabilities |
| market_value | Company's market value |
| asset_turnover | Net Sales / Total Assets |
| total_receivables | Total accounts receivable |
| retained_earnings | Accumulated retained earnings |
| gross_profit | Revenue - Cost of Goods Sold |
| debt_ratio | Total Liabilities / Total Assets |
| depreciation_and_amortization | Depreciation and amortization expenses |
| total_liabilities | Total company liabilities |
| cost_of_goods_sold | Cost of goods sold |
| inventory | Current inventory value |
| ebit | Earnings Before Interest and Taxes |
| current_assets | Total current assets |
| total_current_liabilities | Total current liabilities |
| net_income | Net income |

**Troubleshooting**

If the container fails to start:

Check if port 9696 is available
Ensure Docker is running
Check Docker logs: docker logs bankruptcy-prediction


If predictions fail:

Verify all required features are included in your input
Ensure values are numeric
Check the API endpoint URL

**Model Information**
```text
Type: Random Forest Classifier
Performance: AUC = 0.86 on validation set
Features: 15 financial indicators
Target: Binary classification (0: No Bankruptcy, 1: Bankruptcy)
```