#eda
# Load the required libraries
import pandas as pd
import numpy as np
import seaborn as sns

# Load the data
df = pd.read_csv("C:\\Users\\siddharth\\Documents\\SMR\\ifood_df.csv")

# View the data
df.head()
print(df.head)

# Basic information
df.info()
print(df.info)

# Describe the data
df.describe()
print(df.describe)

# Find the duplicates
df.duplicated().sum()
print(df.duplicated().sum)

# Correlation
df.corr()
print(df.corr())

##CORRELATION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:\\Users\\siddharth\\Documents\\SMR\\ifood_df.csv")
corr = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()

#line plot
import seaborn as sns
import matplotlib.pyplot as plt

# loading dataset
import pandas as pd
data = pd.read_csv("C:\\Users\\siddharth\\Documents\\SMR\\ifood_df.csv")

# draw lineplot
sns.lineplot(x="Age", y="Income", data=data)

# Removing the spines
sns.despine()
plt.show()

#box plot

import matplotlib.pyplot as plt

dataframe = pd.read_csv("C:\\Users\\siddharth\\Documents\\SMR\\ifood_df.csv")

Attribute = dataframe["Income"]


columns = [Attribute]

fig, ax = plt.subplots()
ax.boxplot(columns)
plt.show()

Attribute1 = dataframe["Age"]


columns = [Attribute1]

fig, ax = plt.subplots()
ax.boxplot(columns)
plt.show()

Attribute2 = dataframe["MntTotal"]


columns = [Attribute2]

fig, ax = plt.subplots()
ax.boxplot(columns)
plt.show()

#scatter plot
import matplotlib.pyplot as plt

plt.style.use('seaborn')  # to get seaborn scatter plot

# read the csv file to extract data
data =pd.read_csv("C:\\Users\\siddharth\\Documents\\SMR\\ifood_df.csv")
Income = data['Income']
MntTotal= data['MntTotal']

plt.scatter(Income, MntTotal, s=100, alpha=0.6, edgecolor='black', linewidth=1)

plt.title('Total Expenses vs Income')
plt.xlabel('Income')
plt.ylabel('MntTotal')

plt.tight_layout()
plt.show()
# regression
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
data = pd.read_csv("C:\\Users\\siddharth\\Documents\\SMR\\ifood_df.csv")
x = data["Income"]
y = data["Age"]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()

#using sweetviz library
import pandas as pd
import sweetviz as sv
data = pd.read_csv("C:\\Users\\siddharth\\Documents\\SMR\\ifood_df.csv")
my_report = sv.analyze([data, "DATA"],target_feat='Income')
my_report.show_html('Report.html')

# calculate the Pearson's correlation between two variables
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr
# seed random number generator
seed(1)
# prepare data
data1 = data
corr = pearsonr(data)
print('Pearsons correlation: %.3f' % corr)