from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
data = pd.read_csv("./iris.csv")

# Split data into features and target
X = data.drop(columns="variety")
y = data["variety"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Initialize KNN model with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Predict on test data
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

accuracy, classification_rep


# Set the style for the plots
sns.set(style="whitegrid")

# Pairplot to visualize the relationships between features, colored by the 'variety' class
pairplot = sns.pairplot(data, hue="variety", diag_kind="kde", markers=["o", "s", "D"])
pairplot.fig.suptitle("Pairwise feature relationships by Iris class", y=1.02)

plt.show()


# Adjusting the pairplot to use histograms for diagonal plots
pairplot = sns.pairplot(data, hue="variety", diag_kind="hist", markers=["o", "s", "D"])
pairplot.fig.suptitle("Pairwise feature relationships by Iris class", y=1.02)

plt.show()
