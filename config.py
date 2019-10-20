# coding: utf-8
import torch
import os
from pathlib import Path
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Config(object):
    def __init__(self):
        self.USE_CUDA            = torch.cuda.is_available() # Whether Cuda is available
        self.NET_SAVE_PATH       = Path("./source/trained_net/")   # The path to save the net
        self.DATASET_PATH        = "Your-Task-Name"          # Your task name this time
        self.SUMMARY_PATH        = Path("./source/summary/") / self.DATASET_PATH # Path to dump the tensorboard summary file
        self.TRAINDATARATIO      = 0.7  # Proportion of train images num to test images num
        self.RE_TRAIN            = False # Whether load the trained model or retrain
        self.IS_TRAIN            = True # Whether it is training or predicting
        self.PIC_SIZE            = 256 # The size your image will be resized to before use
        self.NUM_TEST            = 0   # Number of test images
        self.NUM_TRAIN           = 0   # Number of train images  
        self.TOP_NUM             = 1   # Top N to be right(Normally don't use)
        self.NUM_EPOCHS          = 10  # The number of epoches you want to train
        self.BATCH_SIZE          = 16  # Batch size when train
        self.TEST_BATCH_SIZE     = 8   # Batch size when test
        self.NUM_WORKERS         = 4   # Number of worker to load files
        self.NUM_CLASSES         = 2   # Number of classes
        self.LEARNING_RATE       = 0.001 # Learning rate
