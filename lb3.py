import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
file_path = "./iris.csv"
data = pd.read_csv(file_path)

data.head()


X = data[["sepal.length", "sepal.width", "petal.length", "petal.width"]]

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

data["cluster"] = kmeans.labels_

plt.figure(figsize=(8, 6))
plt.scatter(
    data["petal.length"],
    data["petal.width"],
    c=data["cluster"],
    cmap="viridis",
    s=50,
    alpha=0.7,
)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("K-Means Clustering (k=3)")
plt.colorbar(label="Cluster")
plt.show()
