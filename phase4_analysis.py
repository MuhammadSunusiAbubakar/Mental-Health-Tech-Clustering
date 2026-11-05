import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

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

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_cleaned['Cluster'] = kmeans.fit_predict(df_scaled)

print("--- CLUSTER DISTRIBUTION ---")
print(df_cleaned['Cluster'].value_counts().sort_index())

print("\n--- CLUSTER ANALYSIS (Averages for Key Features) ---")
key_features = [col for col in df_cleaned.columns if 'employer' in str(col).lower() or 'treatment' in str(col).lower()]

if key_features:
    cluster_means = df_cleaned.groupby('Cluster')[key_features].mean()
    print(cluster_means.T)
else:
    print("Could not locate key features for targeted HR analysis.")