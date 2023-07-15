import cv2
import numpy as np
from glob import glob
import os
import ast
import math
import progressbar
import pickle as pkl
from numpy.lib import stride_tricks
from skimage import feature
from Code import haralick_
from Code.grid_sampling import SelfAdaption
from Code.Co_training import CoTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('agg')

# initialization
s = 20  # step size
sam_p = 0  # number of positive samples in this round
loss = 1.0  # minimum loss
lo = 1  # loss
flag = 0  # Whether to update the step size
p_n = 0  # sampling number with minimum loss
i = 1  # round i


def read_data(image_dir, label_dir, f=1):

    file_list = glob(os.path.join(image_dir, '*.png'))
    image_list = []
    label_list = []

    for file in file_list:

        image_list.append(cv2.imread(file, f))
        label_list.append(cv2.imread(os.path.join(label_dir, os.path.basename(file).split('.')[0]+'.png'), 0))

    return file_list, image_list, label_list


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

    # print('[INFO] Computing haralick features.')
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

    # print ('[INFO] Computing local binary pattern features.')
    lbp = feature.local_binary_pattern(img, p, r)
    return (lbp-np.min(lbp))/(np.max(lbp)-np.min(lbp)) * 255


def create_features(img, img_gray, label, sam_p=None, train=True):

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
        adaption = SelfAdaption(label, sam_p)
        ss_idx = adaption.cut_adapt(16)

        features = features[ss_idx]
        labels = labels[ss_idx]

    else:
        labels = None

    h_features = harlick_features(img_gray, h_neigh, ss_idx)
    features = np.hstack((features, h_features))

    return features, labels


def create_training_dataset(sam_p, file_list, image_list, label_list, co=False):

    print('[INFO] Creating training dataset on %d image(s).' % len(image_list))

    X = []
    z = []

    for i, img in enumerate(image_list):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features, labels = create_features(img, img_gray, label_list[i], sam_p)
        X.extend(features)
        z.extend(labels)

    X = np.array(X)
    z = np.array(z)

    if co is False:
        z = z.ravel()
        return X, z

    else:

        dataset = np.hstack((X, z))
        np.random.shuffle(dataset)
        X = dataset[:, :13]  # 77
        z = dataset[:, 13].ravel()

        z[z == 255] = 1
        # from collections import Counter
        # print(Counter(z))

        y = np.full(len(z), -1)
        y[y.shape[0] // 2:] = z[z.shape[0] // 2:]

        return X, y


def load_features(img):
    features = []
    with open(feature_path + '{}.txt'.format(os.path.basename(img).split('.')[0])) as f:
        line = f.readlines()
        feature = ' '.join(line)
        feature = feature.split(' ')
        n = len(feature)
        [features.append(ast.literal_eval(feature[i])) for i in range(n)]

    features = np.array(features)
    features = features.reshape((features.shape[0]) // 13, 13)

    return features


def feature_list(image_list, file_list):
    f = []
    sel = np.array([0, 1, 3, 5, 6])  # Select which images to use for loss calculation
    pre_img = np.array(image_list)[sel]
    pre_file = np.array(file_list)[sel]
    for k, (img, file) in enumerate(zip(pre_img, pre_file)):
        print('[INFO] Loading features...')
        features = load_features(file)
        f.append(features)

    return pre_img, pre_file, f


def sam_calc(file_list, image_list, label_list):
    global s
    global sam_p
    global loss
    global lo
    global flag
    global p_n

    # calculate sam_p
    try:
        assert lo <= loss
    except:
        flag = 1
    if flag == 0:
        p_n = sam_p
        loss = lo
        sam_p += s
    else:
        s //= 2
        if s < 5:  # Gradient descent ends when step is less than 5
            if p_n % 20 == 0 and sam_p > p_n:
                sam_p = p_n - 20
                s = 20
                lo = loss
                return False
            else:
                print('Optimal number of positive samples: ', p_n, '，loss:', loss)
                lg_co_clf = training(sam_p, file_list, image_list, label_list)
                pkl.dump(lg_co_clf, open(output_model, "wb"))
                print('Model saved')
        elif lo < loss:
            p_n = sam_p
            loss = lo
            if sam_p > p_n:
                sam_p += s
            else:
                sam_p -= s
        else:
            if sam_p < p_n:
                sam_p += s
            else:
                sam_p -= s


def training(sam_p, file_list, image_list, label_list):
    X_train, y_train = create_training_dataset(sam_p, file_list, image_list,
                                               label_list, co=True)
    X1 = X_train[:, 0:4]
    X2 = np.delete(X_train, 3, axis=1)

    print('[INFO] Training...')
    lg_co_clf = CoTrainingClassifier(RandomForestClassifier(n_estimators=15, max_depth=11, random_state=42))
    lg_co_clf.fit(X1, X2, y_train)

    return lg_co_clf


def tranpixel():
    file_list = glob(os.path.join(pre_out, '*.png'))
    for file in file_list:
        img = cv2.imread(file, 0)
        img[img == 1] = 255
        cv2.imwrite(os.path.join(pre_out, os.path.basename(file)), img)


def loss_calc():

    global lo
    _, image_list, label_list = read_data(pre_out, label_dir, f=0)

    i = []
    j = []
    for m, img in enumerate(image_list):
        label = label_list[m]
        h, w = img.shape
        [i.append(1) for r in range(0, h, 1) for k in range(0, w, 1) if label[r][k] == 255 and img[r][k] == 0]
        [j.append(1) for r in range(0, h, 1) for k in range(0, w, 1) if label[r][k] == 0 and img[r][k] == 255]

    i = len(i)
    j = len(j)
    lo = (i*0.9 + j*0.1)/(262144*5)*100


def main(image_dir, label_dir):

    file_list, image_list, label_list = read_data(image_dir, label_dir)
    pre_img, pre_file, f = feature_list(image_list, file_list)

    while True:

        global s
        global sam_p
        global lo
        global i

        # calculate sam_p
        if sam_calc(file_list, image_list, label_list) == False:
            continue
        elif s < 5:
            break

        # training
        print('[INFO] Training round {}, '.format(i), 'Number of positive samples：', sam_p)
        lg_co_clf = training(sam_p, file_list, image_list, label_list)

        # prediction
        print('[INFO] Prediction round {}'.format(i))
        for j, img in enumerate(pre_img):
            predictions = lg_co_clf.predict(f[j][:, 0:4], np.delete(f[j], 3, axis=1))
            pred_size = int(math.sqrt(f[j].shape[0]))
            inference_img = predictions.reshape(pred_size, pred_size)
            cv2.imwrite(os.path.join(pre_out, os.path.basename(pre_file[j])), inference_img)

        # calculate loss
        tranpixel()
        loss_calc()
        print('[INFO] lo：', lo)
        i += 1


if __name__ == "__main__":

    image_dir = '../../Data/modis/train/'
    label_dir = '../../Data/modis/train_label/'
    pre_out = '../modis/'
    feature_path = 'features/'
    output_model = '../model/our_co.p'
    main(image_dir, label_dir)
