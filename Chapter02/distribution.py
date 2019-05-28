import numpy as np
import pandas as pd
import seaborn as sns;

sns.set()
import matplotlib.pyplot as plt

# initiating random number
np.random.seed(11)

# mean and standard deviation for the x belonging to the first class
mu_x1, sigma_x1 = 0, 0.1

# constat to make the second distribution different from the first
x2_mu_diff, y2_mu_diff = 0.3, 0.6

# creating the first distribution
d1 = pd.DataFrame({'x': np.random.normal(mu_x1, sigma_x1, 1000),
                   'y': np.random.normal(mu_x1, sigma_x1, 1000),
                   'type': 'A'})

# creating the second distribution
d2 = pd.DataFrame({'x': np.random.normal(mu_x1, sigma_x1, 1000) + x2_mu_diff,
                   'y': np.random.normal(mu_x1, sigma_x1, 1000) + y2_mu_diff,
                   'type': 'B'})

data = pd.concat([d1, d2])

ax = sns.scatterplot(x="x", y="y", hue="type",
                     data=data)

plt.show()
