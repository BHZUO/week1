import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE

plt.rcParams['font.sans-serif'] = ['SimHei']

# 加载数据集
df = pd.read_csv('nigerian-songs.csv')

# 选择特征
features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'tempo']

# 检查缺失值
missing_values = df[features].isnull().sum()
print("每个特征的缺失值数量：\n", missing_values)

# 处理缺失值
if missing_values.sum() > 0:
    print("删除包含缺失值的行。")
    df.dropna(subset=features, inplace=True)
else:
    print("未发现缺失值。")

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# 肘部法则
def plot_elbow_method(X_scaled):
    distortions = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 10), distortions, marker='o')
    plt.title('肘部法则')
    plt.xlabel('聚类数')
    plt.ylabel('失真度')
    plt.show()

plot_elbow_method(X_scaled)

# 轮廓系数
def plot_silhouette_scores(X_scaled):
    silhouette_scores = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 10), silhouette_scores, marker='o')
    plt.title('轮廓系数')
    plt.xlabel('聚类数')
    plt.ylabel('轮廓系数')
    plt.show()

plot_silhouette_scores(X_scaled)

# KMeans聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# 层次聚类
agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_labels = agg_clustering.fit_predict(X_scaled)

# 将聚类标签添加到原始数据集中
df['KMeans_Cluster'] = kmeans_labels
df['Agg_Cluster'] = agg_labels

# t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

def plot_clustering_results(X_tsne, labels, title):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette='viridis', legend='full')
    plt.title(title)
    plt.show()

# KMeans聚类结果
plot_clustering_results(X_tsne, df['KMeans_Cluster'], 'KMeans聚类结果（t-SNE可视化）')

# 层次聚类结果
plot_clustering_results(X_tsne, df['Agg_Cluster'], '层次聚类结果（t-SNE可视化）')

# 评估KMeans聚类
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
kmeans_db_score = davies_bouldin_score(X_scaled, kmeans_labels)

# 评估层次聚类
agg_silhouette = silhouette_score(X_scaled, agg_labels)
agg_db_score = davies_bouldin_score(X_scaled, agg_labels)

print(f'KMeans轮廓系数：{kmeans_silhouette}')
print(f'层次聚类轮廓系数：{agg_silhouette}')
print(f'KMeans戴维斯-邦丁指数：{kmeans_db_score}')
print(f'层次聚类戴维斯-邦丁指数：{agg_db_score}')

# 特征分布直方图
df[features].hist(bins=20, figsize=(15, 10), layout=(3, 3))
plt.suptitle('特征分布直方图', fontsize=16)
plt.show()

# 特征相关性热图
correlation_matrix = df[features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('特征相关性热图')
plt.show()

# 特征散点图矩阵
sns.pairplot(df[features], diag_kind='kde', plot_kws={'alpha': 0.5})
plt.suptitle('特征散点图矩阵', fontsize=16)
plt.show()

# 特征箱线图
plt.figure(figsize=(15, 10))
df[features].boxplot()
plt.title('特征箱线图')
plt.show()