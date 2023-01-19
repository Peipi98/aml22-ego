from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import sys
import pickle as pk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os

features = []

def getImage(path, zoom=1):
    return OffsetImage(plt.imread(path), zoom=zoom)


# Function to load features from saved_features folder and cluster them through K-means
def k_means():
    # Load features from saved_features folder
    global features
    for file in os.listdir('./saved_features/dense/test'):
        path = os.path.join('./saved_features/dense/test', file)
        tmp = open(path, 'rb')
        pk_file = pk.load(tmp)
        tmp.close()

        tmp_arr = []
        for feat in pk_file['features']:
            tmp_arr.append(feat['features_RGB'])
        features.append(np.array(tmp_arr))


    # Convert features to numpy array
    features = np.array(features)
    features = features.reshape(-1,features.shape[1]*features.shape[2])
    # Create K-means object
   
    scaled_arr = features

    print(features.shape)
    print(features[0].shape)
    
    print(f'np shape: {np.shape(features)}')
    # Choose features index
    #index = 2
    pca = PCA(2)
    scaled_arr = pca.fit_transform(scaled_arr)

    kmeans = KMeans(n_clusters=8, random_state=42)

    # Get labels
    #labels = kmeans.labels_
    labels_pred = kmeans.fit_predict(scaled_arr)
    #print(labels)
    print(labels_pred)
    # Get centroids
    centroids = kmeans.cluster_centers_
    print(centroids.shape)
    # Create dataframe
    #df = pd.DataFrame()

    c_frames_dict = extract_central_frames()
    image_path = []
    print(c_frames_dict[1])
    for l in c_frames_dict[1]:

        image_path.append(l['path'])
    #print(image_path)

    # Plot clusters using images from test_image_path as markers
    fig, ax = plt.subplots()
    ax.scatter(scaled_arr[:, 0], scaled_arr[:, 1], c=labels_pred, s=0, cmap='viridis')
    #ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=250, alpha=1)
    for x0, y0, path in zip(scaled_arr[:, 0], scaled_arr[:, 1], image_path):
        ab = AnnotationBbox(getImage(path, 0.1), (x0, y0), frameon=False)
        ax.add_artist(ab)

    # Save dataframe to csv
    #df.to_csv('k_means.csv')
    # Plot clusters
    #plt.scatter(scaled_arr[:, 0], scaled_arr[:, 1], c=labels_pred, s=50, cmap='viridis')
    #plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=250, alpha=1)
    plt.show()

def extract_central_frames():
    # Load features from saved_features folder
    splits = []
    for file in os.listdir('./train_val'):
        path = os.path.join('./train_val', file)
        tmp = open(path, 'rb')
        pk_file = pk.load(tmp)
        tmp.close()
        split = []
        print(pk_file)
        for i,line in pk_file.iterrows():
                s = dict()
                s['uid'] = line['uid']
                cframe = (line['start_frame']+line['stop_frame'])//2
                s['cframe'] = cframe
                cframe = f'{cframe}'.zfill(10)
                video_id = line['video_id']
                s['path'] = f'./Data/Epic_Kitchens_reduced/{video_id}/img_{cframe}.jpg'
                split.append(s)

        splits.append(split) # 0 train, 1 test

    return splits

k_means()
