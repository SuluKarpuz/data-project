import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.formula.api import ols
import statsmodels.api as sm

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    print("Original data shape:", df.shape)
    
    df.columns = df.columns.str.strip()
    df['Depression State'] = df['Depression State'].str.strip()
    df['Depression State'] = df['Depression State'].str.replace('\t', '')
    df['Depression State'] = df['Depression State'].str.replace('2', '')
    df['Depression State'] = df['Depression State'].str.replace('5', '')
    
    symptom_cols = [col for col in df.columns if col not in ['Number', 'Depression State']]
    
    for col in symptom_cols:
        df[col] = df[col].apply(lambda x: 5 if x > 5 else x)
        df[col] = df[col].apply(lambda x: np.nan if x < 1 else x)
    
    imputer = SimpleImputer(strategy='median')
    df[symptom_cols] = imputer.fit_transform(df[symptom_cols])
    
    df = df.drop_duplicates()
    df = df[df['Depression State'].notna()]
    df = df[df['Depression State'].isin(['No depression', 'Mild', 'Moderate', 'Severe'])]
    
    print("Cleaned data shape:", df.shape)
    return df

def analyze_feature_importance(df):
    features = [col for col in df.columns if col not in ['Number', 'Depression State']]
    X = df[features]
    
    le = LabelEncoder()
    y = le.fit_transform(df['Depression State'])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lasso = LogisticRegression(penalty='l1', solver='liblinear', multi_class='ovr', C=0.01)
    lasso.fit(X_scaled, y)
    lasso_importance = pd.DataFrame({
        'Feature': features,
        'Lasso_Importance': np.abs(lasso.coef_).mean(axis=0)
    })
    
    rfe = RFE(estimator=LogisticRegression(multi_class='ovr'), n_features_to_select=5)
    rfe.fit(X_scaled, y)
    rfe_importance = pd.DataFrame({
        'Feature': features,
        'RFE_Rank': rfe.ranking_
    })
    
    pca = PCA()
    pca.fit(X_scaled)
    pca_importance = pd.DataFrame({
        'Feature': features,
        'PCA_Importance': np.abs(pca.components_[0])
    })
    
    feature_importance = pd.merge(lasso_importance, rfe_importance, on='Feature')
    feature_importance = pd.merge(feature_importance, pca_importance, on='Feature')
    
    feature_importance['Composite_Score'] = (
        feature_importance['Lasso_Importance'] * 0.4 +
        (1 / feature_importance['RFE_Rank']) * 0.3 +
        feature_importance['PCA_Importance'] * 0.3
    )
    
    return feature_importance.sort_values('Composite_Score', ascending=False)

def visualize_feature_importance(feature_importance, output_file='feature_importance.png'):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.RdYlBu(np.linspace(0.2, 0.8, len(feature_importance)))
    
    bars = plt.barh(
        feature_importance['Feature'],
        feature_importance['Composite_Score'],
        color=colors
    )
    
    plt.title('Feature Importance for Depression Classification', pad=20, size=14)
    plt.xlabel('Composite Importance Score')
    plt.ylabel('Features')
    
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width,
            bar.get_y() + bar.get_height()/2,
            f'{width:.3f}',
            ha='left',
            va='center',
            fontweight='bold'
        )
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_correlation_matrix(df, output_file='feature_correlations.png'):
    features = [col for col in df.columns if col not in ['Number', 'Depression State']]
    correlation_matrix = df[features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='RdYlBu',
        center=0,
        fmt='.2f',
        square=True
    )
    plt.title('Feature Correlation Matrix', pad=20, size=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def perform_descriptive_statistics(df):
    features = [col for col in df.columns if col not in ['Number', 'Depression State']]
    
    desc_stats = df[features].describe()
    print("\nDescriptive Statistics:")
    print(desc_stats)
    
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(features, 1):
        plt.subplot(4, 4, i)
        sns.histplot(data=df, x=col, bins=20)
        plt.title(f'{col} Distribution')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return desc_stats

def perform_chi_square_test(df):
    features = [col for col in df.columns if col not in ['Number', 'Depression State']]
    chi_square_results = {}
    
    for feature in features:
        contingency = pd.crosstab(df['Depression State'], df[feature])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        chi_square_results[feature] = {
            'chi2': chi2,
            'p_value': p_value,
            'dof': dof
        }
    
    return pd.DataFrame(chi_square_results).T

def perform_correlation_analysis(df):
    features = [col for col in df.columns if col not in ['Number', 'Depression State']]
    
    pearson_corr = df[features].corr(method='pearson')
    spearman_corr = df[features].corr(method='spearman')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(pearson_corr, annot=True, cmap='RdYlBu', center=0, fmt='.2f')
    plt.title('Pearson Correlation Matrix', pad=20)
    plt.tight_layout()
    plt.savefig('pearson_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(spearman_corr, annot=True, cmap='RdYlBu', center=0, fmt='.2f')
    plt.title('Spearman Correlation Matrix', pad=20)
    plt.tight_layout()
    plt.savefig('spearman_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return pearson_corr, spearman_corr

def perform_hypothesis_tests(df):
    features = [col for col in df.columns if col not in ['Number', 'Depression State']]
    test_results = {}
    
    for feature in features:
        groups = [group[feature].values for name, group in df.groupby('Depression State')]
        f_stat, p_value = stats.f_oneway(*groups)
        test_results[feature] = {
            'test': 'One-way ANOVA',
            'f_statistic': f_stat,
            'p_value': p_value
        }
    
    for feature in features:
        no_dep = df[df['Depression State'] == 'No depression'][feature]
        severe = df[df['Depression State'] == 'Severe'][feature]
        t_stat, p_value = stats.ttest_ind(no_dep, severe)
        test_results[f'{feature}_t_test'] = {
            'test': 'Independent t-test',
            't_statistic': t_stat,
            'p_value': p_value
        }
    
    return pd.DataFrame(test_results).T

def main():
    df = load_and_clean_data("Deepression.csv")
    desc_stats = perform_descriptive_statistics(df)
    pearson_corr, spearman_corr = perform_correlation_analysis(df)
    chi_square_results = perform_chi_square_test(df)
    print("\nChi-square test results:")
    print(chi_square_results)
    hypothesis_results = perform_hypothesis_tests(df)
    print("\nHypothesis test results:")
    print(hypothesis_results)
    feature_importance = analyze_feature_importance(df)
    print("\nFeature Importance Rankings:")
    print(feature_importance)
    visualize_feature_importance(feature_importance)
    create_feature_correlation_matrix(df)
    return df, desc_stats, chi_square_results, hypothesis_results, feature_importance

if __name__ == "__main__":
    df, desc_stats, chi_square_results, hypothesis_results, feature_importance = main()
