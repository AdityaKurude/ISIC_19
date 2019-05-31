from keras.applications.xception import Xception
from keras.preprocessing import image
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.neighbors import NearestNeighbors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os

DATASET_PATH = "F:\\Ubuntu\\ISIC2019\\Dataset_19\\"

img_width = 718
img_height = 542
img_shape = (img_width, img_height, 3)

config = dict()
config['use_existing'] = True
config['knn_list'] = [2,3,5,7]
config['eps'] = 25200
config['min_pts'] = 5
config['num_images'] = 50
config['find_params'] = False


def extract_features():
    print("Extracting features")
    model = Xception(weights='imagenet', include_top=False, input_shape=img_shape)

    datset = DATASET_PATH + "ISIC_2019_Training_Input\\"
    images = os.listdir(datset)

    feature_list = []

    for img_name in images[:config['num_images']]:
        img = image.load_img(datset + img_name, target_size=(img_width, img_height))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_feature = model.predict(img_data, batch_size=1)
        feature_np = np.array(img_feature)
        feature_list.append(feature_np.flatten())

    feature_list_np = np.array(feature_list)
    print(feature_list_np.shape)
    np.save("xception_features_with_dim_{}x{}".format(img_width, img_height), feature_list_np)


def get_dbscan_param():
    features = np.load("xception_features_with_dim_{}x{}".format(img_width, img_height) + '.npy')
    ns_list = config['knn_list']
    for ns in ns_list:
        nbrs = NearestNeighbors(n_neighbors=ns).fit(features)
        distances, indices = nbrs.kneighbors(features)
        distanceDec = sorted(distances[:, ns - 1], reverse=True)
        # plt.plot(indices[:, 0], distanceDec)
        plt.plot(list(range(1, features.shape[0] + 1)), distanceDec)
        plt.show()


def form_clusters():
    features = np.load("xception_features_with_dim_{}x{}".format(img_width, img_height) + '.npy')
    # db = DBSCAN(eps=config['eps'], min_samples=config['min_pts']).fit(features)
    # labels = db.labels_
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)
    #
    # print('Estimated number of clusters: %d' % n_clusters_)
    # print('Estimated number of noise points: %d' % n_noise_)
    # # pca_features = PCA.fit(n_components=2)
    #
    # plt.scatter(features[:,0], features[:,1],c=db, cmap='Paired')
    # plt.title("DBSCAN")

    clust = OPTICS(min_samples=5, xi=.05)

    # Run the fit
    clust.fit(features)

    space = np.arange(len(features))
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 1)
    ax1 = plt.subplot(G[0, :])
    ax2 = plt.subplot(G[1, 0])

    # Reachability plot
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = space[labels == klass]
        Rk = reachability[labels == klass]
        ax1.plot(Xk, Rk, color, alpha=0.3)
    ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
    ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
    ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
    ax1.set_ylabel('Reachability (epsilon distance)')
    ax1.set_title('Reachability Plot')

    # OPTICS
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = features[clust.labels_ == klass]
        ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    ax2.plot(features[clust.labels_ == -1, 0], features[clust.labels_ == -1, 1], 'k+', alpha=0.1)
    ax2.set_title('Automatic Clustering\nOPTICS')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    if not config['use_existing']:
        extract_features()

    if config['find_params']:
        get_dbscan_param()

    form_clusters()
