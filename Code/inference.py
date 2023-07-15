import cv2
from glob import glob
import os
import pickle as pkl
from Code import train
import math
import matplotlib
# import concurrent.futures
# from slide_crop import slide_crop_a
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
matplotlib.use('agg')
# import time


def create_features(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features, _ = train.create_features(img, img_gray, label=None, file = None, train=False)

    return features


def compute_prediction(file, img, model):

    features = create_features(img)
    # print('Feature Time %.2f seconds' % float(time.time() - start))
    # try:
    #     np.savetxt('{}.txt'.format(os.path.basename(file).split('.')[0]), features)
    #     print('[INFO] {}.txt保存完成'.format(os.path.basename(file).split('.')[0]))
    # except Exception as e:
    #     print('[INFO] {}.txt保存失败'.format(e))
    #     pass
    # finally:
    f = features.reshape(-1, features.shape[1])
    # predictions = model.predict(f)
    predictions = model.predict(f[:, 0:4], np.delete(f, 3, axis=1))  # 13
    # print('Predict Time %.2f seconds' % float(time.time() - start))
    pred_size = int(math.sqrt(features.shape[0]))
    inference_img = predictions.reshape(pred_size, pred_size)

    return inference_img


def infer_images(image_dir, model_path, output_dir):

    filelist = glob(os.path.join(image_dir, '*.png'))

    print('[INFO] Running inference on %s train images' % len(filelist))

    model = pkl.load(open(model_path, "rb"))

    for file in filelist:
        print('[INFO] Processing images:', os.path.basename(file))
        inference_img = compute_prediction(file, cv2.imread(file, 1), model)
        cv2.imwrite(os.path.join(output_dir, os.path.basename(file)), inference_img)


if __name__ == '__main__':
    model_path = "../model/RF.p"
    image_dir = "../Data/modis/test"
    output_dir = "../Data/modis/"
    infer_images(image_dir, model_path, output_dir)
