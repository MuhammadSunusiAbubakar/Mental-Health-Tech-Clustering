import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

file_name = 'mental-health-in-tech-2016.csv'
df = pd.read_csv(file_name)

threshold = len(df) * 0.5
df_cleaned = df.dropna(thresh=threshold, axis=1).copy()

for col in df_cleaned.columns:
    if df_cleaned[col].isnull().sum() > 0:
        mode_val = df_cleaned[col].mode()[0]
        df_cleaned[col] = df_cleaned[col].fillna(mode_val)

gender_col = [col for col in df_cleaned.columns if 'gender' in str(col).lower()][0]
df_cleaned[gender_col] = df_cleaned[gender_col].astype(str).str.lower().str.strip()

male_terms = ['male', 'm', 'man', 'cis male', 'male.', 'mail']
female_terms = ['female', 'f', 'woman', 'cis female', 'femail']

df_cleaned[gender_col] = df_cleaned[gender_col].apply(
    lambda x: 'male' if x in male_terms else ('female' if x in female_terms else 'other')
)

le = LabelEncoder()
for col in df_cleaned.select_dtypes(include=['object', 'str']).columns:
    df_cleaned[col] = df_cleaned[col].astype(str)
    df_cleaned[col] = le.fit_transform(df_cleaned[col])

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cleaned)

pca = PCA(n_components=2)
pca_features = pca.fit_transform(df_scaled)
df_pca = pd.DataFrame(data=pca_features, columns=['PC1', 'PC2'])

print("--- PCA EXPLAINED VARIANCE ---")
print(f"Component 1: {pca.explained_variance_ratio_[0]:.4f}")
print(f"Component 2: {pca.explained_variance_ratio_[1]:.4f}")
print(f"Total Variance Captured: {sum(pca.explained_variance_ratio_):.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(df_pca['PC1'], df_pca['PC2'], alpha=0.5, c='#0052cc')
plt.title('PCA: Mental Health in Tech (2D Projection)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, alpha=0.3)
plt.savefig('pca_visualization.png')
print("\n[SUCCESS] PCA visualization saved as 'pca_visualization.png'")