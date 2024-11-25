# %% [markdown]
# # Project Intro

# %%


# %% [markdown]
# ## Libraries

# %%
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from scipy import stats
import pickle

import warnings
warnings.filterwarnings("ignore")


# %% [markdown]
# # **1. Data Preparation and Data Cleaning**

# %%
df = pd.read_csv('../data/american_bankruptcy.csv')

# %%
df.head()

# %%
df.info()

# %%
# Rename "X" columns

df.rename(columns={
    'X1':'current_assets',
    'X2':'cost_of_goods_sold',
    'X3':'depreciation_and_amortization',
    'X4':'ebitda',
    'X5':'inventory',
    'X6':'net_income',
    'X7':'total_receivables',
    'X8':'market_value',
    'X9':'net_sales',
    'X10':'total_assets',
    'X11':'total_long_term_debt',
    'X12':'ebit',
    'X13':'gross_profit',
    'X14':'total_current_liabilities',
    'X15':'retained_earnings',
    'X16':'total_revenue',
    'X17':'total_liabilities',
    'X18':'total_operating_expenses'}, inplace=True)

df.head()

# %%
df.status_label.value_counts(normalize=True)

# %%
df.status_label = (df.status_label == 'failed').astype(int)
df.status_label.value_counts(normalize=True)

# %%
df.isnull().sum()

# %%
df.rename(columns={'status_label':'bankruptcy'}, inplace=True)
df.head(1).T

# %%


# %%


# %%


# %% [markdown]
# # **2. EDA**

# %%


# %%
def calculate_financial_ratios(df):
    df['year'] = pd.to_datetime(df['year'], format='%Y')
    # Financial ratios
    df['current_ratio'] = df['current_assets'] / df['total_current_liabilities']
    df['debt_ratio'] = df['total_liabilities'] / df['total_assets']
    df['profit_margin'] = df['net_income'] / df['net_sales']
    df['asset_turnover'] = df['net_sales'] / df['total_assets']
    df['ebitda_margin'] = df['ebitda'] / df['net_sales']

    return df

# %%
financial_ratios =  [
    'current_ratio',
    'debt_ratio',
    'profit_margin',
    'asset_turnover',
    'ebitda_margin']

# %%
df_with_ratios = calculate_financial_ratios(df)
df_with_ratios.head(1).T

# %% [markdown]
# ## Range of values

# %% [markdown]
# ### Range of values - Financial Ratios

# %%
df_with_ratios.describe()[financial_ratios]

# %% [markdown]
# ### Range of values - Original variables

# %%
df_with_ratios[df_with_ratios.columns.difference(financial_ratios)].describe()

# %% [markdown]
# ## Missing values

# %%
df_with_ratios.isnull().sum()

# %% [markdown]
# ## Target Variable

# %%
df_with_ratios.groupby('bankruptcy').size().plot(kind='pie',
                                     autopct='%.1f%%',
                                     fontsize=13,
                                     colors=['green', 'blue'])

plt.title('Distribution bankruptcy', size=20)
plt.tight_layout()
plt.show()

print(df_with_ratios['bankruptcy'].value_counts(normalize=True))
print(df_with_ratios['bankruptcy'].value_counts())

# %% [markdown]
# ## Feature Importance

# %%
def analyze_correlations(df, target='bankruptcy'):

    # target 
    y = df[target]
    
    # numerical variables
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != target]
    
    # Calculate correlations
    correlations = {
        'spearman': {},
        'pearson': {}
    }
    
    for col in numeric_cols:
        # Spearman correlation
        spearman_corr, _ = stats.spearmanr(df[col], y)
        correlations['spearman'][col] = spearman_corr
        
        # Pearson correlation
        pearson_corr, _ = stats.pearsonr(df[col], y)
        correlations['pearson'][col] = pearson_corr
    
    # Create DataFrame
    corr_df = pd.DataFrame({
        'spearman_correlation': correlations['spearman'],
        'pearson_correlation': correlations['pearson']
    }).round(3)
    
    # Sort DataFrame by absolute Spearman correlation
    corr_df = corr_df.reindex(
        corr_df['spearman_correlation'].abs().sort_values(ascending=False).index
    )
    
    # Plotting
    plt.figure(figsize=(15,8))
    
    # Correlation comparison plot
    plt.subplot(1, 2, 1)
    plt.scatter(corr_df['pearson_correlation'],
                corr_df['spearman_correlation'],
                alpha=0.6)
    plt.plot([-1, 1], [-1, 1], 'r--')  # diagonal reference line
    plt.xlabel('Pearson Correlation')
    plt.ylabel('Spearman Correlation')
    plt.title('Comparison of Pearson vs Spearman\nCorrelations')
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Top 15 most correlated variables
    plt.subplot(1, 2, 2)
    top_15 = corr_df.head(15)
    
    # Plot both correlations
    x = np.arange(len(top_15))
    width = 0.35
    
    plt.barh(x + width/2, top_15['spearman_correlation'], width, 
             label='Spearman', alpha=0.8)
    plt.barh(x - width/2, top_15['pearson_correlation'], width,
             label='Pearson', alpha=0.8)
    
    plt.yticks(x, top_15.index)
    plt.xlabel('Correlation')
    plt.title('Top 15 Most Correlated Variables\n(Sorted by Spearman)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return corr_df

# %%
corr_results = analyze_correlations(df)

# %%
corr_results

# %% [markdown]
# ## Feature Selection 
# 
# using logistic regression and random forest
# 

# %%
X = df.drop(columns=['company_name', 'bankruptcy', 'year'])
y = df['bankruptcy']

# %%
logit_model = LogisticRegression()
logit_model.fit(X, y)
logit_feature_importance = pd.Series(logit_model.coef_[0], index=X.columns).abs()
logit_feature_importance = logit_feature_importance.sort_values(ascending=False)
logit_feature_importance

# %%
scaler = RobustScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns
)

# %%
logit_model_scaled = LogisticRegression(random_state=42)
logit_model.fit(X_scaled, y)

# %%
# 4. Calcular y ordenar importancia de features
logit_feature_importance_scaled = pd.DataFrame({
    'feature': X.columns,
    'importance': abs(logit_model.coef_[0])
})
logit_feature_importance_scaled = logit_feature_importance_scaled.sort_values('importance', ascending=False)
logit_feature_importance_scaled

# %%
rf_model_scaled = RandomForestClassifier(random_state=42)
rf_model_scaled.fit(X_scaled, y)

# %%
rf_feature_importance_scaled = pd.DataFrame({
    'feature': X.columns,
    'importance': abs(rf_model_scaled.feature_importances_)
})
rf_feature_importance_scaled = rf_feature_importance_scaled.sort_values('importance', ascending=False)
rf_feature_importance_scaled

# %% [markdown]
# ## Feature Selection base on combine importance

# %%
def combine_feature_importance(logit_feature_importance_scaled, rf_feature_importance_scaled, n_features=None):
    # Normalize importances for each model (0-1)
    logit_max = logit_feature_importance_scaled['importance'].max()
    rf_max = rf_feature_importance_scaled['importance'].max()
    
    # Create combine DF
    combined_importance = pd.DataFrame({
        'feature': logit_feature_importance_scaled['feature'],
        'logit_norm': logit_feature_importance_scaled['importance'] / logit_max,
        'rf_norm': rf_feature_importance_scaled['importance'] / rf_max
    })
    
    # Calculate average of normalized importances
    combined_importance['mean_importance'] = (
        combined_importance['logit_norm'] + combined_importance['rf_norm']
    ) / 2
    
    # Sort by average importance
    combined_importance = combined_importance.sort_values(
        'mean_importance', ascending=False
    )
    
    # If n_features is specified, select the top n
    if n_features:
        combined_importance = combined_importance.head(n_features)
        
    return combined_importance

# %%
combined_features = combine_feature_importance(
    logit_feature_importance_scaled,
    rf_feature_importance_scaled,
    n_features=15
)

# %%
combined_features

# %% [markdown]
# # **3. Model Selection**

# %%
final_features = combined_features.feature.to_list()

# %%
df_with_ratios.head(2)

# %% [markdown]
# ## Train-Val-Test Split

# %%
# Conditions for splitting the data

train_condition = df_with_ratios['year'] <= pd.to_datetime('2011')
val_condition = (df_with_ratios['year'] >= pd.to_datetime('2012')) & (df_with_ratios['year']<= pd.to_datetime('2014'))
test_condition = df_with_ratios['year'] > pd.to_datetime('2014')

# %%
df_train = df_with_ratios[train_condition]
df_val = df_with_ratios[val_condition]
df_test = df_with_ratios[test_condition]

print(df_train.shape)
print(df_val.shape)
print(df_test.shape)
print(df_with_ratios.shape)
print(df_train.shape[0]+df_val.shape[0]+df_test.shape[0])

# %%
print("Proportion of data per set:")
print(f"Training: {(train_condition).mean():.2%}")
print(f"Validation: {(val_condition).mean():.2%}")
print(f"Test: {(test_condition).mean():.2%}")
print(f"Not assigned: {(~(train_condition | val_condition | test_condition)).mean():.2%}")


print("\nIntersection between sets (must be 0):")
print(f"Train-Val: {(train_condition & val_condition).sum()}")
print(f"Train-Test: {(train_condition & test_condition).sum()}")
print(f"Val-Test: {(val_condition & test_condition).sum()}")

# %%
X_train = df_train[final_features]
X_val = df_val[final_features]
X_test = df_test[final_features]

y_train = df_train['bankruptcy']
y_val = df_val['bankruptcy']
y_test = df_test['bankruptcy']

# %%
X_train

# %% [markdown]
# ## Random Forest Classifier

# %%
def train_optimize_rf(X_train, X_val, y_train, y_val, random_state=42):
    # define grid params
    param_grid = {
        'n_estimators':[10, 50, 100, 200, 400],
        'max_depth':[3, 5, 10, 15],
        'min_samples_leaf':[1, 3, 5, 10, 50]
    }
    
    # Initialize variables to save best results
    best_auc = 0
    best_params = None
    best_model = None
    
    results = []
    
    # Grid Search
    for n_est in param_grid['n_estimators']:
        for depth in param_grid['max_depth']:
            for min_leaf in param_grid['min_samples_leaf']:
                
                rf = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=depth,
                    min_samples_leaf=min_leaf,
                    random_state=random_state,
                    n_jobs=-1
                )
                rf.fit(X_train, y_train)
                
                y_pred = rf.predict_proba(X_val)[:, 1]
                
                auc = roc_auc_score(y_val, y_pred)
                
                results.append({
                    'n_estimators':n_est,
                    'max_depth':depth,
                    'min_samples_leaf':min_leaf,
                    'auc':auc
                })
                
                if auc > best_auc:
                    best_auc = auc
                    best_params = {
                        'n_estimators':n_est,
                        'max_depth':depth,
                        'min_samples_leaf':min_leaf
                    }
                    best_model = rf
                    
    results_df = pd.DataFrame(results)
    
    results_df = results_df.sort_values('auc', ascending=False)
    # Print best results
    print("\nBest 5 parameter combinations:")
    print(results_df.head())
    
    print(f"\nBest AUC found: {best_auc:.4f}")
    print("\nBest Params:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # Plots
    plt.figure(figsize=(15, 5))
    
    # Plot1 1: AUC vs n_estimators
    plt.subplot(131)
    sns.lineplot(data=results_df, x='n_estimators', y='auc', ci=None,
                )  
    plt.title('AUC vs n_estimators')
    plt.grid(True, linestyle='--', alpha=0.7)  
    
    # Plot 2: AUC vs max_depth
    plt.subplot(132)
    
    plot_df = results_df.copy()
    max_depth_values = plot_df['max_depth'].unique()
    max_depth_values = [d for d in max_depth_values if d is not None]
    if None in plot_df['max_depth'].values:
        plot_df['max_depth'] = plot_df['max_depth'].fillna(max(max_depth_values) + 2)
    
    sns.lineplot(data=plot_df, x='max_depth', y='auc', ci=None,
                )
    if None in results_df['max_depth'].values:
        
        xticks = plt.gca().get_xticks()
        xticks_labels = [str(int(x)) if x != max(max_depth_values) + 2 else 'None' 
                        for x in xticks]
        plt.gca().set_xticklabels(xticks_labels)
    plt.title('AUC vs max_depth')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: AUC vs min_samples_leaf
    plt.subplot(133)
    sns.lineplot(data=results_df, x='min_samples_leaf', y='auc', ci=None,
                )
    plt.title('AUC vs min_samples_leaf')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'best_model': best_model,
        'best_params': best_params,
        'best_auc': best_auc,
        'all_results': results_df
    }

# %%
results_rf = train_optimize_rf(X_train, X_val, y_train, y_val)
best_rf = results_rf['best_model']

# %% [markdown]
# ## XGBoost Classifier

# %%
def train_optimize_xgb(X_train, X_val, y_train, y_val, random_state=42):
   
   # Definir grid de parámetros
   param_grid = {
       'eta': [0.01, 0.05, 0.1, 0.3],  # learning rate
       'max_depth': [3, 5, 7, 10, 15],
       'min_child_weight': [1, 10, 15, 30]
   }
   
   # Inicializar variables para guardar mejores resultados
   best_auc = 0
   best_params = None
   best_model = None
   
   # Para guardar todos los resultados
   results = []
   
   # Grid search manual
   for eta in param_grid['eta']:
       for depth in param_grid['max_depth']:
           for min_child in param_grid['min_child_weight']:
               
               # Crear y entrenar modelo
               xg = xgb.XGBClassifier(
                   eta=eta,
                   max_depth=depth,
                   min_child_weight=min_child,
                   random_state=random_state,
                   n_estimators=100,  
                   use_label_encoder=False,
                   eval_metric='auc',
                   objective = 'binary:logistic'
                   
               )
               
               # Entrenar modelo
               xg.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
               
               # Predecir probabilidades
               y_pred_proba = xg.predict_proba(X_val)[:, 1]
               
               # Calcular AUC
               auc = roc_auc_score(y_val, y_pred_proba)
               
               # Guardar resultados
               results.append({
                   'eta': eta,
                   'max_depth': depth,
                   'min_child_weight': min_child,
                   'auc': auc
               })
               
               # Actualizar mejor modelo si mejora AUC
               if auc > best_auc:
                   best_auc = auc
                   best_params = {
                       'eta': eta,
                       'max_depth': depth,
                       'min_child_weight': min_child
                   }
                   best_model = xg
   
   # Convertir resultados a DataFrame
   results_df = pd.DataFrame(results)
   
   # Ordenar resultados por AUC
   results_df = results_df.sort_values('auc', ascending=False)
   
   # Imprimir mejores resultados
   print("\nMejores 5 combinaciones de parámetros:")
   print(results_df.head())
   
   print(f"\nMejor AUC encontrado: {best_auc:.4f}")
   print("\nMejores parámetros:")
   for param, value in best_params.items():
       print(f"{param}: {value}")
   
   # Crear visualizaciones
   plt.figure(figsize=(15, 5))
   
   # Gráfico 1: AUC vs eta
   plt.subplot(131)
   sns.lineplot(data=results_df, x='eta', y='auc', ci=None,
               marker='o')
   plt.title('AUC vs Learning Rate (eta)')
   plt.grid(True, linestyle='--', alpha=0.7)
   
   # Gráfico 2: AUC vs max_depth
   plt.subplot(132)
   sns.lineplot(data=results_df, x='max_depth', y='auc', ci=None,
               marker='o')
   plt.title('AUC vs max_depth')
   plt.grid(True, linestyle='--', alpha=0.7)
   
   # Gráfico 3: AUC vs min_child_weight
   plt.subplot(133)
   sns.lineplot(data=results_df, x='min_child_weight', y='auc', ci=None,
               marker='o')
   plt.title('AUC vs min_child_weight')
   plt.grid(True, linestyle='--', alpha=0.7)
   
   plt.tight_layout()
   plt.show()
   
   return {
       'best_model': best_model,
       'best_params': best_params,
       'best_auc': best_auc,
       'all_results': results_df
   }


# %%
results_xgb = train_optimize_xgb(X_train, X_val, y_train, y_val)
best_xgb = results_xgb['best_model']

# %%
best_xgb

# %% [markdown]
# ## Selecting the final model

# %%
best_rf.fit(X_train, y_train)
y_pred = best_rf.predict_proba(X_val)[:,1]
roc_auc_score(y_val, y_pred)

# %%
best_xgb.fit(X_train, y_train)
y_pred = best_xgb.predict_proba(X_val)[:,1]
roc_auc_score(y_val, y_pred)

# %%
y_test.value_counts()

# %%
def compare_models(best_rf, best_xgb, X_val, y_val):   
   # Random Forest predictions
   rf_pred_proba = best_rf.predict_proba(X_val)[:, 1]
   rf_auc = roc_auc_score(y_val, rf_pred_proba)
   
   # XGBoost Predictions
   xgb_pred_proba = best_xgb.predict_proba(X_val)[:, 1]
   xgb_auc = roc_auc_score(y_val, xgb_pred_proba)
   
   # Comopare Results
   print("Results on validation set:")
   print(f"Random Forest AUC: {rf_auc:.4f}")
   print(f"XGBoost AUC: {xgb_auc:.4f}")
   
   # Plot ROC Curve
   plt.figure(figsize=(8, 6))
   
   # RF Curve
   fpr_rf, tpr_rf, _ = roc_curve(y_val, rf_pred_proba)
   plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_auc:.4f})')
   
   # XGB Curve
   fpr_xgb, tpr_xgb, _ = roc_curve(y_val, xgb_pred_proba)
   plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {xgb_auc:.4f})')
   
   
   plt.plot([0, 1], [0, 1], 'r--')
   
   plt.xlabel('FP Rate')
   plt.ylabel('TP Rate')
   plt.title('Roc Curves - Model Comparison\n(Validation Set)')
   plt.legend()
   plt.grid(True, linestyle='--', alpha=0.7)
   plt.show()
   
   return {
       'rf_auc': rf_auc,
       'xgb_auc': xgb_auc,
       'rf_pred_proba': rf_pred_proba,
       'xgb_pred_proba': xgb_pred_proba,
       'rf_roc': (fpr_rf, tpr_rf),
       'xgb_roc': (fpr_xgb, tpr_xgb)
   }


# %%
final_results = compare_models(best_rf, best_xgb, X_val, y_val)

# %% [markdown]
# Random Forest is the best model!

# %% [markdown]
# ## Save the model

# %%
output_file = f"model_C=1.0.bin"
output_file

# %%
with open('../models/' + output_file, 'wb') as f_out:
    pickle.dump(best_rf, f_out)

# %%
test_company = {
    'current_ratio': 1.8053735778505688,
    'market_value': 2210.904,
    'asset_turnover':  0.6082410428335904,
    'total_receivables':  562.927,
    'retained_earnings':  1756.044,
    'gross_profit':  924.51,
    'debt_ratio': 0.35913914527339424,
    'depreciation_and_amortization':  916.318,
    'total_liabilities':  1964.443,
    'cost_of_goods_sold':  2402.487,
    'inventory':  65.579,
    'ebit': -125.879,
    'current_assets':  950.197,
    'total_current_liabilities':  526.316,
    'net_income':  -321.421
}

# %%
test_company_df = pd.DataFrame([test_company])

# %%
best_rf.predict_proba(test_company_df)

# %%



