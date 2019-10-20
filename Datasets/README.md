# Folder to store your dataset

1. When you want to start a new project with a different dataset, please make a new directory which name is your task name, like "wash_anime_images" or "Japanese_western". We note it as `Your-Task-Name` here.
2. Enter this new directory, and make directories named `train_data` and `pred_data` here.
3. Assume you have three classes named "A","B","C", then you create three directories here with the same name.
4. Paste all of your classified training images to the corresponding folder.
5. Paste all of your "ready to be predicted" images(The fucking messy data you want to classify) to `pred_data`.
5. Copy your new task name `Your-Task-Name` and open the `config.py` file in the root directory of this project, replace the value of `  DATASET_PATH  ` with this name like `  self.DATASET_PATH = "wash_anime_images"  ` , then save the change.
6. You are all set with your data.