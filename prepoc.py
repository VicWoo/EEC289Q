# This module is to import data from the given folder containing testing data, testing data, and corresponding labels in labels.csv, and categorize all the images based on their breeds. Two new folders train_new and test_test will be created which contain classified images in its sub-folders named by the breeds

import shutil
import os
import pandas as pd

data_dir       = "data" 
old_train_path = 'train'
new_train_path = 'train_new'
old_test_path  = 'test'
new_test_path  = 'for_test'
folder      = 'images_folder'
t=os.path.join('.',data_dir, 'labels.csv')


df = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
old_test_path = os.path.join(data_dir, old_test_path)
new_test_path = os.path.join(data_dir, new_test_path)
if os.path.exists(new_test_path):
    shutil.rmtree(new_test_path)
os.makedirs(new_test_path)

if os.path.exists(os.path.join(new_test_path, folder)):
    shutil.rmtree(os.path.join(new_test_path, folder))
os.makedirs(os.path.join(new_test_path, folder))

for test_file in os.listdir(old_test_path):
    shutil.copy(os.path.join(old_test_path, test_file),
                os.path.join(new_test_path, folder))



