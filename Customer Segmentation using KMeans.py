import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder,MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


df = pd.read_csv('datasets/Mall_Customers.csv')


print(df.info())

print(df.describe())

df.hist()

encoder = OrdinalEncoder()
df['Gender'] = encoder.fit_transform(df[['Gender']])

plt.figure(figsize=(10,6))
plt.title('Correlation Matrix')
sns.heatmap(df.corr(),cmap='cividis',annot=True)

plt.figure(figsize=(15,6))
plt.title('Null Values')
sns.heatmap(df.isnull(),cbar=False,yticklabels=False)
plt.show()

x = np.array(df.drop('CustomerID',axis=1))
print(x.shape)


scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

pca = PCA(n_components=2)
x_transformed = pca.fit_transform(x_scaled)

    
clusters = 5
model = KMeans(n_clusters=clusters,init='k-means++',random_state=42)
preds = model.fit_predict(x_transformed)

df['Cluster'] = preds

print(df.head())