import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from mxnet import image as mximg

# With the following, the output of plotting commands is displayed inline 
# within frontends like the Jupyter notebook, directly below the code cell 
# that produced it. The resulting plots will then also be stored in the 
# notebook document.
#matplotlib inline



data_dir  = "data"
train_dir = "train"
test_dir  = "test"

train_num = len([name for name in os.listdir(os.path.join('.', data_dir, train_dir)) \
    if os.path.isfile(os.path.join(data_dir, train_dir, name))])
test_num = len([name for name in os.listdir(os.path.join('.', data_dir, test_dir)) \
    if os.path.isfile(os.path.join(data_dir, test_dir, name))])

print ("Number of training images: %d" % train_num)
print ("Number of testing timages: %d" % test_num)

labels = pd.read_csv(os.path.join('.', data_dir, "labels.csv"))

print ("Number of classes: %d" % len(set(labels.breed)))
print ("Missing labels: " + str(labels.isnull().values.any()))

class_freq = labels.breed.value_counts()

print(class_freq.head())

print(class_freq.tail())

# Source: https://www.kaggle.com/jeru666/dog-eat-dog-world-eda-useful-scripts
yy = pd.value_counts(labels['breed'])
#print(yy.index)
fig, ax = plt.subplots()
fig.set_size_inches(15, 9)
sns.set_style("whitegrid")


ax = sns.barplot(x = yy.index, y = yy, data = labels)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize = 8)
ax.set(xlabel='Dog Breed', ylabel='Count')
ax.set_title('Distribution of Dog breeds')
plt.show()

rand_idx  = np.random.randint(len(labels))
img_id    = labels.iloc[rand_idx].id
img_class = labels.iloc[rand_idx].breed
img       = mximg.imread(os.path.join('.', data_dir, train_dir, img_id + ".jpg"))

print ("Image shape: ", img.shape)
print ("Image type : ", img_class)
print (img[:2,:2, 0])
plt.imshow(img.asnumpy())
plt.show()


