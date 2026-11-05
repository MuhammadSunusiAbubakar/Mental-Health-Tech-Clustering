import pandas as pd
import matplotlib.pyplot as plt
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

inertia_values = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia_values, marker='o', linestyle='-', color='#0052cc')
plt.title('K-Means Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.grid(True, alpha=0.3)
plt.xticks(k_range)
plt.savefig('elbow_curve.png')

print("--- K-MEANS INERTIA RESULTS ---")
for k, inertia in zip(k_range, inertia_values):
    print(f"k={k}: Inertia = {inertia:.2f}")
print("\n[SUCCESS] Elbow curve saved as 'elbow_curve.png'")