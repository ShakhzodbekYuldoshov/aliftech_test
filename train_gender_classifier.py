import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib
import pandas as pd


label_encoder = LabelEncoder()

# load dataset
train_dataset = pd.read_csv('./dataset/train.csv')
test_dataset = pd.read_csv('./dataset/test.csv')

# drop class column because it is not necessary for us (I thought so)
train_dataset.drop('class', inplace=True, axis=1)
test_dataset.drop('class', inplace=True, axis=1)

# separate features and label
y_train = train_dataset['gender']
X_train = train_dataset.loc[:, train_dataset.columns != 'gender']

y_test = test_dataset['gender']
X_test = test_dataset.loc[:, test_dataset.columns != 'gender']

# categorical to numerical data transformation
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)

# parameters for randoomized search cv
params = {
    'learning_rate': [0.01, 0.1, 0.15, 0.2, 0, 25, 0, 3],
    'max_depth': [2, 3, 5, 8, 10, 12, 15, 18],
    'min_child_weight': [1, 2, 3, 5, 6, 8],
    'gamma': [0, 0.1, 0.2, 0, 3]
}

model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                          colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
                          gamma=0.1, gpu_id=-1, importance_type=None,
                          interaction_constraints='', learning_rate=0.2, max_delta_step=0,
                          max_depth=10, min_child_weight=6,
                          monotone_constraints='()', n_estimators=100, n_jobs=16,
                          num_parallel_tree=1, predictor='auto', random_state=0,
                          reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                          tree_method='exact', validate_parameters=1, verbosity=None)

# used to find best hyperparams to pass into xgb classifier
# random_search = RandomizedSearchCV(
#     model, param_distributions=params, n_iter=5, scoring='roc_auc', n_jobs=-1, cv=5, verbose=3)

trained_model = model.fit(X_train, y_train)

preds = trained_model.predict(X_test)

print('Accuracy:  ', accuracy_score(y_test, preds))
print('Model 1 XGboost Report %r' % (classification_report(y_test, preds)))
# print('Confusion matrix:  ', confusion_matrix(y_test, preds))

trained_model.save_model('./models/best.json')

# joblib.dump(random_search.best_estimator_, './models/model1.json')
# trained_model.save('./datasets/model1.pkl')

# print(random_search.best_estimator_)
