import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou
from train import create_dir, load_data

# image size
H = 256
W = 256

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (H, W, 3)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return ori_x, x                                ## (1, 256, 256, 3)
  
  
def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (H, W)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.int32)                    ## (256, 256)
    return ori_x, x

def save_results(ori_x, ori_y, y_pred, save_image_path):
    line = np.ones((H, 10, 3)) * 255

    ori_y = np.expand_dims(ori_y, axis=-1)  ## (256, 256, 1)
    ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1) ## (256, 256, 3)

    y_pred = np.expand_dims(y_pred, axis=-1)  ## (256, 256, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) ## (256, 256, 3)

    cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred*255], axis=1)
    cv2.imwrite(save_image_path, cat_images)
    
    
if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Folder for saving results """
    create_dir("results_DLV3SA_inception")

    """ Load the model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
        model = tf.keras.models.load_model("files_DLV3SA/model_DLV3SA_inception.h5")

    """ Load the test data """
    dataset_path = "D:/medical_challenge/segmentation/CVC-612/data/"
    # dataset_path = "D:/medical_challenge/segmentation/2d/VIP_CUP_5fold/fold_1/train/"
    # dataset_path = "D:/medical_challenge/segmentation/CVC-612/CVC_ClinicDB"
    # dataset_path = "D:/medical_challenge/segmentation/2d/data_mbrain_5fold/fold_0/train/"
    (train_x, train_y), (valid_x, valid_y) = load_data(dataset_path)
    # print(len(test_x), len(test_y))
    print("=====================================================================================================")
    print(len(valid_x), len(valid_y))
    print("=====================================================================================================")
    print(len(train_x), len(train_y))
    SCORE = []
