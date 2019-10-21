# coding: utf-8
import os
import math
import random
import shutil
import numpy as np
import time
import re
import pickle
import warnings
import argparse
from config import Config
from pathlib import Path
from PIL import Image


warnings.filterwarnings("ignore")


def load_data(root='./Datasets/'):
    """
    :Outputs:
        train_pairs : the path of the train  images and their labels' index list
        test_pairs  : the path of the test   images and their labels' index list
        class_names : the list of classes' names
    :param root : the root location of the dataset.
    """
    IMG_PATH = [root / 'train_data/', root / 'test_data/']
    NAME_PATH = './source/reference/names.pkl'

    class_names = pickle.load(open(NAME_PATH, 'rb'))

    # Cope with train data
    files = []
    train_dirs = [IMG_PATH[0] / i for i in next(os.walk(IMG_PATH[0]))[1]]
    [files.extend([i / j for j in next(os.walk(i))[2] if Path(j).suffix.lower() in [".jpg", ".png"]]) for i in train_dirs]
    train_pairs = [(i, class_names.index(i.parts[-2])) for i in files]

    # Cope with test data
    files = []
    test_dirs = [IMG_PATH[1] / i for i in next(os.walk(IMG_PATH[1]))[1]]
    [files.extend([i / j for j in next(os.walk(i))[2] if Path(j).suffix.lower() in [".jpg", ".png"]]) for i in test_dirs]
    test_pairs = [(i, class_names.index(i.parts[-2])) for i in files]

    return train_pairs, test_pairs

def load_pred_data(root='./Datasets/'):
    IMG_PATH = root / 'pred_data'
    files = [IMG_PATH / i for i in next(os.walk(IMG_PATH))[2] if i.split('.')
                   [-1].lower() in ["jpg", "png"]]
    return files

def load_class_name():
    NAME_PATH = './source/reference/names.pkl'
    class_names = pickle.load(open(NAME_PATH, 'rb'))
    return class_names

def sep_data(opt):
    """
    When there is only a train_data folder, this function will divide these images randomly in to
    train_data and test_data, while the ration is determined by a parameter defined in opt.
    :param opt: the config object of this project.
    :return: None
    """
    train_path = './Datasets'/Path(opt.DATASET_PATH)/'train_data'
    dirs = [train_path/i for i in next(os.walk(train_path))[1]]

    for dirn in dirs:
        files = [dirn/file for file in next(os.walk(dirn))[2] if Path(file).suffix.lower() in [".jpg", ".png"]]
        file_num = len(files)
        if file_num > 0:
            test_part = random.sample(files, math.floor(
                file_num*(1-opt.TRAINDATARATIO)))
            test_path = Path('./Datasets')/opt.DATASET_PATH/'test_data'/dirn.parts[-1]
            if not os.path.exists(test_path):
                os.makedirs(test_path)
            for file in test_part:
                shutil.move(str(file), str(test_path))

            names = [test_path/name for name in os.listdir(test_path) if not name.startswith('.')]
            names.sort(key=lambda x: int(os.path.splitext(os.path.basename(x).split('_')[-1])[0]))
            for i in range(len(names)):
                os.rename(names[i], test_path/(str(i) + names[i].suffix))

            names = [dirn/name for name in os.listdir(dirn) if not name.startswith('.')]
            names.sort(key=lambda x: int(os.path.splitext(os.path.basename(x).split('_')[-1])[0]))
            for i in range(len(names)):
                os.rename(names[i], dirn/(str(i) + names[i].suffix))


def check_img(opt, path=None):
    """
    Check all of the images whether there is something wrong and it cannot be opened.
    Also, if a image has less than 3 channels or it is not a jpg file, it will be removed.
    :param path:
    :return:None
    """
    if path is None:
        path = Path('./Datasets/' + opt.DATASET_PATH + '/train_data/')
    
    dirs = [path/i for i in next(os.walk(path))[1]]
    if len(dirs)==0:
        dirs = [path]
    for dirn in dirs:
        files = [dirn/file for file in next(os.walk(dirn))
                 [2] if file.split('.')[-1].lower() in ["jpg", "png"]]
        file_num = len(files)
        if file_num > 0:
            for filen in files:
                try:
                    print("==> Checking File: ", filen)
                    with Image.open(filen) as img:
                        imgnp = np.array(img)
                        filename, suffix = os.path.splitext(filen)
                        if suffix == '.gif':
                            os.remove(filen)
                        elif len(imgnp.shape) != 3 or imgnp.shape[0] < 3:
                            os.remove(filen)
                        elif suffix not in [".jpg", ".png"]:
                            img.save(filename+'.jpg')
                            os.remove(filen)
                except OSError:
                    os.remove(filen)
                    print("==> File", filen, "Removed!")
        rename_folder(dirn, prefix=opt.DATASET_PATH)


def coalesce_dirs(opt, path='./Datasets/'):
    """
    This function aims to extract all of the jpg image files, rename them, and then move them to the root path.
    It must meet the requirement that there are only dirs in this path and there can only be images with the suffix jpg.
    :param path: The folder you want to coalesce.
    :return: None
    """
    path = path/opt.DATASET_PATH/'train_data'
    dirs = [path/i for i in next(os.walk(path))[1]]
    for i, dirn in enumerate(dirs):
        files = [dirn/file for file in next(os.walk(dirn))[2] if file.suffix.lower() in [".jpg", ".png"]]
        if len(files) > 0:
            rename_folder(dirn, prefix=str(i))
    for i, dirn in enumerate(dirs):
        files = [dirn/file for file in next(os.walk(dirn))[2] if file.suffix.lower() in [".jpg", ".png"]]
        if len(files) > 0:
            for filen in files:
                shutil.move(filen, path + filen.name)
            shutil.rmtree(dirn)
    rename_folder(path)


def re_sep(opt):
    """
    This function aims to redo the separation of dataset images after some arrangement of it like data-washing.
    :param opt:the config object of this project.
    :param root:the place where train_dataset and test_dataset are stored.
    :return:None
    """
    train_path = './Datasets'/Path(opt.DATASET_PATH)/'train_data'
    test_path = './Datasets'/Path(opt.DATASET_PATH)/'test_data'
    train_dirs = [train_path/i for i in next(os.walk(train_path))[1]]
    test_dirs = [test_path/i for i in next(os.walk(test_path))[1]]
    for dirn in train_dirs:
        rename_folder(dirn, prefix='0')
    for dirn in test_dirs:
        rename_folder(dirn)
        files = [dirn/file for file in next(os.walk(dirn))[2] if file.split('.')[-1].lower() in ["jpg", "png"]]
        if len(files) > 0:
            for filen in files:
                shutil.move(filen, train_path/filen.parts[-2]/filen.parts[-1])
    for dirn in train_dirs:
        rename_folder(dirn, prefix='')
    sep_data(opt)


def rename_folder(path, prefix=''):
    """
    Rename all of the image files in a certain path in the format "`prefix`_i.jpg"
    :param path: the path in which the process is implemented.
    :param prefix: the prefix you want to add to all of the files' new name.
    :return: None
    """
    # pattern = re.compile('\d*')

    files = [path/name for name in os.listdir(path) if not name.startswith('.')]
    for i in range(len(files)):
        os.rename(files[i], path/('temp_'+str(i)+ files[i].suffix))
    for i in range(len(files)):
        os.rename(path/('temp_' + str(i) + files[i].suffix),
                  path/(prefix + '_' + str(i) + files[i].suffix))


def gen_name(opt, path='./Datasets/', out_path='./source/reference/names.pkl'):
    """
    Generate a file which contains all of the categories the dataset has.
    :return:
    """
    path = path + opt.DATASET_PATH + '/'
    dirs = []
    dataset_pathes = [path + i + "/" for i in next(os.walk(path))[1]]
    for i, in_path in enumerate(dataset_pathes):
        dirs.extend([i.strip('/') for i in next(os.walk(in_path))[1]])
    classes = list(set(dirs))
    pickle.dump(classes, open(out_path, 'wb'))


def folder_init(opt):
    """
    Initialize folders required
    """
    if not os.path.exists('source'):
        os.mkdir('source')
    if not os.path.exists('source/reference'):
        os.mkdir('source/reference')
    if not os.path.exists('source/summary'):
        os.mkdir('source/summary')
    if not os.path.exists(opt.NET_SAVE_PATH):
        os.mkdir(opt.NET_SAVE_PATH)
    if not os.path.exists(opt.NET_SAVE_PATH / opt.DATASET_PATH):
        os.mkdir(opt.NET_SAVE_PATH / opt.DATASET_PATH)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--check_img', action='store_true', help="Check your dateset images.")
    parser.add_argument('--sep_data', action='store_true', help="Divide your dataset into train and test parts.")
    parser.add_argument('--coalesce_dirs', action='store_true', help="Coalesce your train and test dirs together.")
    parser.add_argument('--re_sep', action='store_true', help="Reseperate your dataset when you make some changes in your dataset folder.")
    args = parser.parse_args()
    opt = Config()
    if args.check_img:
        check_img(opt)
    elif args.sep_data:
        sep_data(opt)
    elif args.coalesce_dirs:
        coalesce_dirs(opt)
    elif args.re_sep:
        re_sep(opt)
    else:
        exit(0)