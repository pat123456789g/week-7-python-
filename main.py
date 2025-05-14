"""
Iris Dataset Analysis
Author: [Your Name]
Date: [Submission Date]
"""

# ========== IMPORT LIBRARIES WITH ERROR HANDLING ==========
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import load_iris
except ImportError as e:
    print(f"Critical Missing Dependency: {e}")
    exit(1)

# ========== DATA LOADING & PREPROCESSING ==========
def load_iris_dataset():
    """Load and prepare Iris dataset with error handling"""
    try:
        # Load dataset from sklearn
        raw_data = load_iris()
        df = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
        
        # Add species names
        df['species'] = [raw_data.target_names[t] for t in raw_data.target]
        
        # Validate dataset
        if df.isnull().sum().sum() > 0:
            raise ValueError("Dataset contains missing values")
        if df.empty:
            raise ValueError("Empty dataset loaded")
            
        return df
    
    except Exception as e:
        print(f"Data Loading Error: {e}")
        exit(1)

# Load dataset
iris_df = load_iris_dataset()
print("✅ Dataset successfully loaded\n")

# ========== EXPLORATORY DATA ANALYSIS ==========
print("=== DATA EXPLORATION ===")
try:
    # Basic information
    print("\n1. Dataset Structure:")
    iris_df.info()
    
    # Statistical summary
    print("\n2. Statistical Overview:")
    print(iris_df.describe())
    
    # Class distribution
    print("\n3. Species Distribution:")
    print(iris_df['species'].value_counts())

except Exception as e:
    print(f"Exploration Error: {e}")

# ========== DATA VISUALIZATION ==========
# Configure plot styling
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.dpi': 120,
    'axes.titlesize': 14,
    'axes.titlepad': 15,
    'axes.labelsize': 12
})

# ----- Visualization 1: Pairwise Relationships -----
try:
    print("\nGenerating Pair Plot...")
    pair_grid = sns.pairplot(
        iris_df,
        hue='species',
        height=2.2,
        plot_kws={'s': 30, 'alpha': 0.8, 'edgecolor': 'k'},
        diag_kind='kde'
    )
    pair_grid.fig.suptitle("Pairwise Feature Relationships by Species", y=1.02)
    plt.show()
    print("""
    Pair Plot Insights:
    - Clear separation between Setosa and other species
    - Strong linear relationship in petal measurements
    - Versicolor/Virginica overlap in sepal dimensions""")
    
except Exception as e:
    print(f"Visualization Error: {e}")

# ----- Visualization 2: Feature Distribution -----
try:
    print("\nGenerating Distribution Plots...")
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    features = iris_df.columns[:-1]
    
    for ax, feature in zip(axs.flat, features):
        sns.violinplot(
            data=iris_df,
            x='species',
            y=feature,
            ax=ax,
            inner='quartile',
            palette='Set2'
        )
        ax.set_title(f"{feature.split(' ')[0].title()} Distribution")
        ax.set_xlabel('')
    
    plt.suptitle("Feature Distributions by Species", y=1.02)
    plt.tight_layout()
    plt.show()
    print("""
    Distribution Insights:
    - Setosa has unique petal characteristics
    - Virginica shows widest feature ranges
    - Sepal width has significant species overlap""")
    
except Exception as e:
    print(f"Visualization Error: {e}")

# ----- Visualization 3: Correlation Analysis -----
try:
    print("\nGenerating Correlation Heatmap...")
    plt.figure(figsize=(8, 6))
    corr_matrix = iris_df.drop('species', axis=1).corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        center=0,
        fmt=".2f",
        linewidths=0.5
    )
    plt.title("Feature Correlation Matrix")
    plt.show()
    print("""
    Correlation Insights:
    - Petal features show near-perfect correlation (0.96)
    - Sepal length correlates with petal features
    - Sepal width shows negative correlations""")
    
except Exception as e:
    print(f"Visualization Error: {e}")

# ========== KEY FINDINGS SUMMARY ==========
print("\n=== FINAL ANALYSIS SUMMARY ===")
print("""
1. Species Differentiation:
   - Setosa: Distinctly smaller petals, wider sepals
   - Versicolor: Intermediate characteristics
   - Virginica: Largest floral dimensions

2. Diagnostic Features:
   - Petal Length/Width: Most discriminative features
   - Sepal Length: Useful for Versicolor/Virginica separation
   - Sepal Width: Least discriminative feature

3. Data Quality:
   - No missing values detected
   - Perfect class balance (50 samples each)
   - Consistent measurement units (cm)

4. Actionable Insights:
   - Petal measurements should be primary classification features
   - Dimensionality reduction possible due to high correlations
   - Potential for high-accuracy classification model""")

# ========== SYSTEM CHECKS ==========
try:
    assert len(iris_df) == 150, "Row count mismatch"
    assert iris_df['species'].nunique() == 3, "Unexpected species count"
    print("\n✅ All system checks passed")
except AssertionError as e:
    print(f"\n⚠️ Validation Warning: {e}")
