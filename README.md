# User Guide of Convenient Image Classification

## Work Flow

1. Go to the `Datasets` folder, and the README.md inside will tell you how to prepare your data.
2. Make sure you have python3 and pytorch installed on your machine. If you want to use GPU to accelerate your training and testing process, you have to install CUDA on your computer too.
3. Open a terminal and switch into this project directory.
4. Do pre-process for your data. Your data might be dirty, containing file types other than jpg or png, or even broken. So please run `python utils.py --check_img`
5. Then, we need to divide the dataset into training set and testing set. Please run `python utils.py --sep_data`. Also, the proportion can be changed in the config.py file, the `TRAINDATARATIO` option. If you want to manually divide your training and testing set, or you want the training set your classified-images and the testing set your images waiting to classify, PLEASE do not run this command.
6. Start training. Run `python main.py`
7. Start prediction. After preparing your "prediction data", please set the `IS_TRAIN` option in `config.py` to **False**. Then run `python main.py` again.
8. Now if there is no bugs, you have already gotten a result file in `source` folder, like `Your-Task-Name_results.pkl`
9. If you only want to get this classification result, then you are done; if you want to actually move different categories of images to different folders, then please run `python move_prediction_files.py`. Then you can check the `./Datasets/Your-Task-Name/Classified` folder to pick up your classified images. 
10. Have a good day!

## Attention

1. The default training epoch is set to 10, which is not enough for completely fit the resnet, if you have good device and are willing to wait for a better result, you can change it to a larger number, like 30~50.
2. The more training data you get, the easier you can fit the net. Also, you'd better have a clear standard of your classification, or the result may not be pretty good.
3.  If you don't understand the code itself, please don't try to make arrangement to the code or the configure file, or the program may not work properly.