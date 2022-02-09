import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import pandas as pd


# load and preprocess dataset
test_dataset = pd.read_csv('./dataset/test.csv')
test_dataset.drop('class', inplace=True, axis=1)

# separate features and label
y_test = test_dataset['gender']
X_test = test_dataset.loc[:, test_dataset.columns != 'gender']

label_encoder = LabelEncoder()
y_test = label_encoder.fit_transform(y_test)

# initialize and load model
model = xgb.XGBClassifier()
model.load_model('./models/best_gc.json')

# predict the model
preds = model.predict(X_test)

print('Accuracy:  ', accuracy_score(y_test, preds))
print('Model 1 XGboost Report %r' % (classification_report(y_test, preds)))
print('Confusion matrix:  ', confusion_matrix(y_test, preds))

plot_confusion_matrix(model, X_test, y_test)
plt.savefig('confusion_matrix.png')
plt.show()
