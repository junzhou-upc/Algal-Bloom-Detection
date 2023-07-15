import cv2
import numpy as np
from glob import glob
import os
import progressbar
import pickle as pkl
from numpy.lib import stride_tricks
from skimage import feature
from Code import haralick_
import matplotlib
from Code.grid_sampling import SelfAdaption

matplotlib.use('agg')
# from sklearn.model_selection import train_test_split
# import random
# import time


def read_data(image_dir, label_dir):

    print('[INFO] Reading image data.')

    filelist = glob(os.path.join(image_dir, '*.png'))
    image_list = []
    label_list = []

    for file in filelist:

        image_list.append(cv2.imread(file, 1))
        label_list.append(cv2.imread(os.path.join(label_dir, os.path.basename(file).split('.')[0]+'.png'), 0))

    return filelist, image_list, label_list


def subsample_idx(low, high, sample_size):

    return np.random.randint(low, high, sample_size)


def calc_haralick(roi):

    feature_vec = []

    if roi.all() == 0:
        texture_features = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    else:
        texture_features = haralick_.haralick(roi, return_mean=True)
    # mean_ht = texture_features.mean(axis=0)

    [feature_vec.append(i) for i in texture_features[0:9]]

    return np.array(feature_vec)


def harlick_features(img, h_neigh, ss_idx):

    print('[INFO] Computing haralick features.')
    size = h_neigh
    shape = (img.shape[0] - size + 1, img.shape[1] - size + 1, size, size)
    strides = 2 * img.strides
    patches = stride_tricks.as_strided(img, shape=shape, strides=strides)
    patches = patches.reshape(-1, size, size)

    if len(ss_idx) == 0:
        bar = progressbar.ProgressBar(maxval=len(patches),
                                      widgets=[progressbar.Bar('=', '[', ']'),
                                               ' ', progressbar.Percentage()])
    else:
        bar = progressbar.ProgressBar(maxval=len(ss_idx),
                                      widgets=[progressbar.Bar('=', '[', ']'),
                                               ' ', progressbar.Percentage()])

    bar.start()

    h_features = []

    if len(ss_idx) == 0:
        for i, p in enumerate(patches):
            bar.update(i+1)
            h_features.append(calc_haralick(p))
    else:
        for i, p in enumerate(patches[ss_idx]):
            bar.update(i+1)
            h_features.append(calc_haralick(p))

    # h_features = [calc_haralick(p) for p in patches[ss_idx]]

    return np.array(h_features)


def create_binary_pattern(img, p, r):

    print ('[INFO] Computing local binary pattern features.')
    lbp = feature.local_binary_pattern(img, p, r)
    return (lbp-np.min(lbp))/(np.max(lbp)-np.min(lbp)) * 255


def create_features(img, img_gray, label, train=True):

    lbp_radius = 24  # local binary pattern neighbourhood
    h_neigh = 11  # haralick neighbourhood
    # num_examples = 2000  # number of examples per image to use for training model

    lbp_points = lbp_radius*8

    feature_img = np.zeros((img.shape[0], img.shape[1], 4))
    feature_img[:, :, :3] = img
    img = None
    feature_img[:, :, 3] = create_binary_pattern(img_gray, lbp_points, lbp_radius)
    features = feature_img.reshape(feature_img.shape[0]*feature_img.shape[1], feature_img.shape[2])

    border = 5  # (haralick neighbourhood - 1) / 2

    img_gray = cv2.copyMakeBorder(img_gray, top=border, bottom=border,
                                  left=border, right=border,
                                  borderType=cv2.BORDER_CONSTANT,
                                  value=[0, 0, 0])
    ss_idx = []
    if train is True:

        labels = label.reshape(label.shape[0]*label.shape[1], 1)
        # [ss_idx.append(i) for i in range(len(labels)) if labels[i]==255]
        # ss_idx = random.sample(ss_idx, 150)
        # ss_idx = np.append(np.array(ss_idx), subsample_idx(0, features.shape[0], 1500))

        adaption = SelfAdaption(label, 55)
        ss_idx = adaption.cut_adapt(4)
        # _, ss_idx = adaption.adapt()

        features = features[ss_idx]
        labels = labels[ss_idx]

    else:
        labels = None

    h_features = harlick_features(img_gray, h_neigh, ss_idx)
    features = np.hstack((features, h_features))

    return features, labels


def create_training_dataset( filelist, image_list, label_list):

    print('[INFO] Creating training dataset on %d image(s).' % len(image_list))

    X = []
    z = []

    for i, img in enumerate(image_list):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features, labels = create_features(img, img_gray, label_list[i], filelist[i])
        X.extend(features)
        z.extend(labels)

    X = np.array(X)
    # X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
    z = np.array(z)
    # z = z.reshape(z.shape[0]*z.shape[1], z.shape[2])

    dataset = np.hstack((X, z))
    np.random.shuffle(dataset)
    X = dataset[:, :13]  # 77
    z = dataset[:, 13].ravel()

    z[z == 255] = 1
    # from collections import Counter
    # print(Counter(z))

    y = np.full(len(z), -1)
    y[y.shape[0]//2:] = z[z.shape[0]//2:]

    X_test = X[-X.shape[0]//10:]
    y_test = y[-y.shape[0]//10:]

    X_l = X[X.shape[0]//2:-X.shape[0]//10]
    y_l = y[y.shape[0]//2:-y.shape[0]//10]

    X_train = X[:-X.shape[0]//10]
    y_train = y[:-y.shape[0]//10]

    print('[INFO] Feature vector size:', X_train.shape)

    return X_train, X_test, y_train, y_test, X_l, y_l


from sklearn.ensemble import RandomForestClassifier
from Code.Co_training import CoTrainingClassifier
# from sklearn.metrics import classification_report
# import time


def main(image_dir, label_dir, output_model):

    filelist, image_list, label_list = read_data(image_dir, label_dir)
    X_train, X_test, y_train, y_test, X_l, y_l = create_training_dataset( filelist, image_list, label_list)

    X1 = X_train[:, 0:4]
    X2 = np.delete(X_train, 3, axis=1)

    print('RandomForestClassifier CoTraining')
    lg_co_clf = CoTrainingClassifier(RandomForestClassifier(n_estimators=15, max_depth=11, random_state=42))
    lg_co_clf.fit(X1, X2, y_train)
    pkl.dump(lg_co_clf, open(output_model, "wb"))
    # y_pred = lg_co_clf.predict(X_test[:, 0:4], np.delete(X_test, 3, axis=1))
    # print(classification_report(y_test, y_pred))
    # print('Our Training Time %.2f seconds' % float(time.time()-start))


if __name__ == "__main__":
    image_dir = '../Data/modis/train/'
    label_dir = '../Data/modis/train_label/'
    # classifier = 'RF'
    output_model = '../model/our_co.p'
    main(image_dir, label_dir, output_model)
