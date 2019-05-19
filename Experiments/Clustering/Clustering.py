from keras.applications.xception import Xception
from keras.preprocessing import image
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os

DATASET_PATH = "F:\\Ubuntu\\ISIC2019\\Dataset_19\\"

img_width = 718
img_height = 542
img_shape = (img_width, img_height, 3)

config = dict()
config['is_existing'] = False
config['knn_list'] = [2,5,7,8,10,12,20,30]
config['eps'] = 2
config['min_pts'] = 2
config['num_images'] = 100

def extract_features():
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
    db = DBSCAN(eps=config['eps'], min_samples=config['min_pts']).fit(features)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    # pca_features = PCA.fit(n_components=2)

    plt.scatter(features[:,0], features[:,1],c=db, cmap='Paired')
    plt.title("DBSCAN")


if __name__ == "__main__":

    if not config['is_existing']:
        extract_features()

    get_dbscan_param()
    form_clusters()
