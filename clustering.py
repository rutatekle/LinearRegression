import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator


def preprocessing(dataset):
    scaler = StandardScaler()
    return scaler.fit_transform(dataset)


def clustering(k, x, x_target):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x)
    labels = kmeans.labels_
    centeriod = kmeans.cluster_centers_

    print("\nCluster centers")
    print(centeriod)

    print("\nTarget Vs Predicted Labels")
    print(pd.crosstab(x_target, labels))

    plt.scatter(x[:, 0], x[:, 1], cmap='rainbow')
    plt.title("Dataset plot")
    plt.show()

    y_kmeans = kmeans.predict(x)
    error = kmeans.inertia_

    print("\n SSE is " + str(error))

    plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=18, color='red')
    plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=18, color='green')
    plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=18, color='blue')

    plt.scatter(centeriod[0][0], centeriod[0][1], marker='*', s=200, color='Black')
    plt.scatter(centeriod[1][0], centeriod[1][1], marker='*', s=200, color='Black')
    plt.scatter(centeriod[2][0], centeriod[2][1], marker='*', s=200, color='Black')
    plt.title("Clusters plot")
    plt.show()


def predict_k_elbow():
    Error = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i).fit(x)
        kmeans.fit(x)
        Error.append(kmeans.inertia_)

    plt.plot(range(1, 11), Error)
    plt.xlabel('No of clusters')
    plt.ylabel('SSE')
    plt.title("SSE Vs No of clusters : Elbow method")
    plt.show()
    return KneeLocator(range(1, 11), Error, curve="convex", direction="decreasing")


dataset = pd.read_csv("https://utd-class.s3.amazonaws.com/clustering/seeds_dataset.csv")

scaled_features = preprocessing(dataset)
x = scaled_features[:, :6]
x_target = dataset.iloc[:, -1]
k = 3
clustering(k, x, x_target)

kl = predict_k_elbow()
print("\nOptimum number of k using Elbow method is : " + str(kl.elbow))
