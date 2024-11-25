import requests

# test data
test_company = {
    'current_ratio': 1.8053735778505688,
    'market_value': 2210.904,
    'asset_turnover': 0.6082410428335904,
    'total_receivables': 562.927,
    'retained_earnings': 1756.044,
    'gross_profit': 924.51,
    'debt_ratio': 0.35913914527339424,
    'depreciation_and_amortization': 916.318,
    'total_liabilities': 1964.443,
    'cost_of_goods_sold': 2402.487,
    'inventory': 65.579,
    'ebit': -125.879,
    'current_assets': 950.197,
    'total_current_liabilities': 526.316,
    'net_income': -321.421
}

# URL
url = 'http://localhost:9696/predict'

# make requests
response = requests.post(url, json=test_company)
print('Status code:', response.status_code)
print('Prediction:', response.json())