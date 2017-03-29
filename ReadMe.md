# Big Data Analytics
## Homework 1 (M10515053 陳品陵)
### Question & Answer

1.	哪些屬性對於惡意程式分類有效？
    - 利用 Feature importance with scikit-learn Random Forest 所算出的 importances 值愈**高**者。
2.	哪些屬性對於惡意程式分類無效？
    - 利用 Feature importance with scikit-learn Random Forest 所算出的 importances 值愈**低**者。
3.	用什麼方法可以幫助你決定上述的結論？
    - Feature importance with scikit-learn Random Forest。
4.	透過Python哪些套件以及方法可以幫助你完成上面的工作？
    - pandas
    - numpy
    - matplotlib
    - Algo: Feature importance with scikit-learn Random Forest
5.	課程迄今有無建議？(老師教學風趣或良好就不用再提囉)
    - 在學習這堂課的過程需要一些先備知識，感謝老師都有在課堂上提供一些 Tutorial，讓我們更容易上手，不過還是有點希望老師的教學步調可以再放慢些。

### Implementation
- Development environment: Ubuntu、Ipython2.0
- Dataset: https://github.com/ManSoSec/Microsoft-Malware-Challenge
- Reference Code: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

### My Code
**>>Input**
```  python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

mydata = pd.read_csv('/home/ally/文件/LargeTrain.csv')

X = np.array(mydata.ix[:,0:1804])
y = np.array(mydata.ix[:,1804:1805]).ravel()

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=10,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(10):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
# Choose top 10
plt.bar(range(10), importances[indices[0:10]], color="orange")
plt.show()
```
**>>Output**
![png](HW1_output.png)