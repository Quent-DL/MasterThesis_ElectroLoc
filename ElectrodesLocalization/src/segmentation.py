from sklearn.cluster import KMeans
import numpy as np


def segment_electrodes(contacts, n_electrodes):
    # TODO: extract features
    features = contacts

    kmeans = KMeans(n_clusters=n_electrodes)
    labels = kmeans.fit_predict(features)

    # TODO create the electrodes based on the labels
    electrodes = np.array([contacts])

    return electrodes