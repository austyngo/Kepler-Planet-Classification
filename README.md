# Kepler Planet Classification
Classifying observed objects from the Kepler mission with scikit-learn.

 The NASA Exoplanet Science Institute keeps an archive of observed objects from the Kepler Mission that may possibly be habitable planets. This dataset contains 9564 observations from this archive, with various characteristics and a label to classify them. The classification, named koi_disposition, sorts each observations into one of three classes: candidate, confirmed, and false positive. Using this dataset, we will train a model to predict the disposition class of each observation based on its features.
 
 Data source: [NASA](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=koi)
 
## Data Preprocessing
Drop error and additional identification features.
```python 
#regular expression to identify columns labeled with 'err'(for error)
df_disp = df[df.columns.drop(df.filter(regex='err'))]
df_disp = df_disp.drop(['koi_pdisposition', 'koi_score', 'kepoi_name', 'kepler_name', 'koi_tce_delivname', 'koi_tce_plnt_num'],
                        axis=1).set_index('kepid')

#drop rows with null values
df_disp=df_disp.dropna() 
```

Setting features as X and target variable as y
```python
# koi_disposition is the target variable
y = df_disp['koi_disposition']
#everything remaing except koi_disposition are feature variables
X = df_disp.loc[:, df_disp.columns != 'koi_disposition']
```
Before fitting the dataset to a model, we should check for any highly correlated features. This can be done using a correlation matrix.
```python
#plot correlation matrix
import matplotlib.pyplot as plt

plt.figure(figsize=(17,8))
cor = X.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.RdGy)
plt.show()
```
![correlation matrix](https://github.com/austyngo/Kepler-Planet-Classification/blob/master/images/corr.png)

It looks like there are a few pairs of features that are highly correlated:

  1. koi_slogg and koi_srad
  2. koi_insol and koi_srad
  3. koi_impact and koi_prad
  4. koi_depth and koi_model_snr
We will drop one feature from each pair: koi_srad (in two pairs), koi_prad, and koi_depth
```python
X = X.drop(['koi_srad', 'koi_prad', 'koi_depth'], axis=1)
```

There is a rather large range of values among the dataset, which can cause the classification model to assign incorrect weights to certain features. To account for this, the features will be standardized by removing the mean and scaling to unit variance.
```python
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
mapper = DataFrameMapper([(X.columns, StandardScaler())])
scaled_features = mapper.fit_transform(X)
X = pd.DataFrame(scaled_features, index=X.index, columns=X.columns)
```
Checking if there is a target feature imbalance that may affect training the model.
```python
y.value_counts()
```
FALSE POSITIVE    4582
CONFIRMED         2355
CANDIDATE         2263
Name: koi_disposition, dtype: int64

There looks to be a reasonable spread of target classes. We will not need to over/undersample the dataset.

## Model Training and Parameter Selection
Splitting the dataset into training and test sets.
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```
Using Grid Search Cross Validation to identify the optimal parameters for the Random Forest classification model.
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

# Number of trees in random forest
n_estimators = [5, 10, 20, 30]
# Maximum number of levels in tree
max_depth = range(2,20)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = range(1,5)

param_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

grid_clf = GridSearchCV(clf, param_grid, cv=8, scoring='accuracy')
grid_clf.fit(X_train, y_train)
```
GridSearchCV(cv=8, estimator=RandomForestClassifier(),
             param_grid={'max_depth': range(2, 20),
                         'min_samples_leaf': range(1, 5),
                         'min_samples_split': [2, 5, 10],
                         'n_estimators': [5, 10, 20, 30]},
             scoring='accuracy')
```python
bscore = grid_clf.best_score_
print(f"The best accuracy score found by the Grid Search is {round(bscore * 100, 2)}%")
bparams = grid_clf.best_params_
print(f"The Random Forest paramaters that return the best accuracy score is:\n {bparams}")
```
The best accuracy score found by the Grid Search is 90.28%

The Random Forest paramaters that return the best accuracy score is:
 {'max_depth': 17, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 30}
 
 Applying the best paramaters into a final model.
 ```python
 best_params = {'max_depth': 13,
                'min_samples_leaf': 2,
                'min_samples_split': 2,
                'n_estimators': 30}

clf_best = RandomForestClassifier()
clf_best.set_params(**best_params)
clf_best.fit(X_train, y_train)
pred_best = clf_best.predict(X_test)
```

## ROC AUC Evaluation
Another method of evaluating the model is to display the receiver operating characteristic (ROC) curve and the area under the curve (AUC). This method plots a grapth that shows the performance of the classification model at every classification threshold by plotting two metrics, the True Positive Rate and the False Positive Rate. In this case, since there are three classes in the target variable, three ROC curves are plotted, one for each class.

Since we have a multi-class target variable (CONFIRMED, CANDIDATE, FALSE POSITIVE), we will need to binarize the class labels to plot the ROC curve. Each class with be evaluated separately with its own curve plotted.
```python
y_bin = label_binarize(y, classes=['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'])
n_classes = y_bin.shape[1]

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_bin, test_size= 0.33, random_state=1)
```
```python
from sklearn.multiclass import OneVsRestClassifier
classifier = OneVsRestClassifier(clf_best)
y_score = classifier.fit(X_train2, y_train2).predict_proba(X_test2)
```
```python
from sklearn.metrics import roc_curve, auc
from itertools import cycle

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test2[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=1.5, label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i+1, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k-', lw=1.5)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for Kepler Identification')
plt.legend(loc="lower right")
plt.show()
```
![ROC AUC](https://github.com/austyngo/Kepler-Planet-Classification/blob/master/images/roc_auc.png)

### Interpreting the ROC AUC
From the plotted curves, we can see that the Random Forest classifier performs very well in identifying the Kepler distposition, as they nearly reach the top left corner of the chart.

The AUC, or the area under the ROC curve, measures the entire area beneath each curve. The greater the area, the better performing the model. This can be used to compare models or in this case, compare the precision of our classifier model on each target class. The AUC score ranges from 0 to 1, with 0 indicating a model predicting everything wrong and 1 indicating that a model predicts the target perfectly.

The AUC can be interpreted as the probability that our model selects a random actual positive example over a random negative example when predicting positive outcomes. Class 3 (green) is predicted perfectly with a score of 1 and classes 1 (blue) and 2 (red) having near perfect scores of 0.96 and 0.95 respectively.

Now we will plot the actual confusion matrix with the model's predictions.
```python
from sklearn.metrics import confusion_matrix

clf_best_2 = RandomForestClassifier()
clf_best_2.set_params(**best_params)
clf_best_2.fit(X_train, y_train)

pred_best = clf_best_2.predict(X_test)

labels = ['CONFIRMED', 'FALSE POSITIVE', 'CANDIDATE']
cm = confusion_matrix(y_test, pred_best, labels)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
```
[705, 3, 92]

[0, 1466, 13]

[183 , 6 , 568]

![Confusion Matrix](https://github.com/austyngo/Kepler-Planet-Classification/blob/master/images/conf.png)
