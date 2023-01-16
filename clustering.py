
from sklearn.cluster import KMeans
import numpy as np
import sys
import pickle as pk
import pandas as pd
import matplotlib.pyplot as plt
import os

features = []

# Function to load features from saved_features folder and cluster them through K-means
def k_means():
    # Load features from saved_features folder
    global features
    for file in os.listdir('./saved_features/dense/train'):
        path = os.path.join('./saved_features/dense/train', file)
        tmp = open(path, 'rb')
        pk_file = pk.load(tmp)
        tmp.close()

        tmp_arr = []
        for feat in pk_file['features']:
            tmp_arr.append(feat['features_RGB'][2])
        features.append(np.array(tmp_arr))


    # Convert features to numpy array
    features = np.array(features)
    # Create K-means object
    
    print(features.shape)
    print(features[0].shape)
    
    print(f'np shape: {np.shape(features)}')
    kmeans = KMeans(n_clusters=8, random_state=0).fit(features[0])

    # Get labels
    labels = kmeans.labels_
    labels_pred = kmeans.fit_predict(features[0])
    print(labels)
    print(labels_pred)
    # Get centroids
    centroids = kmeans.cluster_centers_
    print(centroids.shape)
    # Create dataframe
    #df = pd.DataFrame({'labels': labels, 'centroids': centroids})

    # Save dataframe to csv
    #df.to_csv('k_means.csv')

    # Plot clusters
    plt.scatter(features[0][:, 0], features[0][:, 1], c=labels_pred, s=50, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=250, alpha=1)
    plt.show()



k_means()
