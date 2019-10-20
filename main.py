# coding: utf-8
import torchvision.transforms as transforms
import torchvision.models as models
import sys
import pickle

from torch.utils.data import DataLoader
from utils import *
from data_loader import *
from train import *
from config import Config

def main():
    opt = Config()

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((opt.PIC_SIZE, opt.PIC_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((opt.PIC_SIZE, opt.PIC_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    folder_init(opt)
    gen_name(opt)
    class_names = load_class_name()
    opt.NUM_CLASSES = len(class_names)

    net = models.resnet152(pretrained=True)
    fc_features = net.fc.in_features
    net.fc = nn.Linear(fc_features, opt.NUM_CLASSES)

    if opt.IS_TRAIN:
        train_pairs, test_pairs = load_data(Path("./Datasets") / opt.DATASET_PATH)
        trainDataset = TRAIN_LOADER(train_pairs, opt, transform_train)
        train_loader = DataLoader(dataset=trainDataset, batch_size=opt.BATCH_SIZE, shuffle=True, num_workers=opt.NUM_WORKERS, drop_last=False)

        testDataset  = TRAIN_LOADER(test_pairs, opt, transform_test)
        test_loader  = DataLoader(dataset=testDataset,  batch_size=opt.TEST_BATCH_SIZE, shuffle=False, num_workers=opt.NUM_WORKERS, drop_last=False)

        opt.NUM_TRAIN    = len(trainDataset)
        opt.NUM_TEST     = len(testDataset)

        model_name = opt.NET_SAVE_PATH / opt.DATASET_PATH / ('%s_model.pkl' % str(net.__class__.__name__))
        print("==> Using: ", model_name)
        if os.path.exists(model_name) and not opt.RE_TRAIN:
            net = torch.load(model_name)

        net = training(opt, train_loader, test_loader, net, class_names)

    else:
        check_img(opt, path=Path("./Datasets") / opt.DATASET_PATH / 'pred_data')
        predData = load_pred_data(Path("./Datasets") / opt.DATASET_PATH)
        predDataset = PRED_LOADER(predData, transform_test)
        pred_loader = DataLoader(dataset=predDataset,  batch_size=opt.TEST_BATCH_SIZE, shuffle=False, num_workers=opt.NUM_WORKERS, drop_last=False)
        model_name = opt.NET_SAVE_PATH / opt.DATASET_PATH / ('%s_model.pkl' % str(net.__class__.__name__))
        print("==> Using: ", model_name)
        if os.path.exists(model_name):
            net = torch.load(model_name)
            print("Load existing model: %s" % model_name)
            results = predicting(opt, pred_loader, net, class_names)
            out_name = './source/%s_results.pkl'% str(opt.DATASET_PATH)
            pickle.dump(results, open(out_name, 'wb'))
            print("==> Prediction finished! Result file is at ", out_name, ". Please run `python move_prediction_files.py` if you need.")
        else:
            try:
                sys.exit(0)
            except:
                print("Error!You haven't trained your net while you try to test it.")
            finally:
                print('Program stopped.')
            

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()