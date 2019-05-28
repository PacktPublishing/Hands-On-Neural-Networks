import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense
from sklearn.metrics import roc_auc_score
# from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
import seaborn as sns;

sns.set()

# initiating random number
np.random.seed(11)

#### Creating the dataset

# mean and standard deviation for the x belonging to the first class
mu_x1, sigma_x1 = 0, 0.1

# constat to make the second distribution different from the first
x2_mu_diff = 0.35

# creating the first distribution
d1 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1 , 1000),
                   'x2': np.random.normal(mu_x1, sigma_x1 , 1000),
                   'type': 0})

# creating the second distribution
d2 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1 , 1000) + x2_mu_diff,
                   'x2': np.random.normal(mu_x1, sigma_x1 , 1000) + x2_mu_diff,
                   'type': 1})

data = pd.concat([d1, d2], ignore_index=True)


ax = sns.scatterplot(x="x1", y="x2", hue="type",
                      data=data)

# Splitting the dataset in training and test set
msk = np.random.rand(len(data)) < 0.8

# Roughly 80% of data will go in the training set
train_x, train_y = data[['x1', 'x2']][msk], data.type[msk]
# Everything else will go into the validation set
test_x, test_y = data[['x1', 'x2']][~msk], data.type[~msk]

my_perceptron = Sequential()
my_perceptron.add(Dense(1, input_dim=2, init="zero", activation="linear"))
my_perceptron.compile(loss="mse", optimizer=SGD(lr=0.01))
my_perceptron.fit(train_x.values, train_y, nb_epoch=1, batch_size=1, shuffle=False)


pred_y = my_perceptron.predict(test_x)

print(roc_auc_score(test_y, pred_y))
