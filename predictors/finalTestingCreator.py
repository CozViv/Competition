import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, KBinsDiscretizer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.svm import SVC

testing_flag = True

impute_flag = True
binning_flag = False

# Importing data
training_data = pd.read_csv("../ml-2024-f/train_final.csv")
testing_data = pd.read_csv("../ml-2024-f/test_final.csv")

# Dropping duplicate data
training_data.drop("education", axis=1)
testing_data.drop("education", axis=1)

# Splitting into features and labels
features = training_data.drop("income>50K", axis=1)
labels = training_data["income>50K"]

scaled_features = ['fnlwgt','capital.gain', 'capital.loss', 'age','fnlwgt', 'hours.per.week']
imputed_featuers = ['capital.gain', 'capital.loss']
categorical_features = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
binned_features = ['age','fnlwgt', 'hours.per.week']

scaler = StandardScaler()
encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
# imputer = KNNImputer(missing_values=0)
imputer = SimpleImputer(missing_values=0, strategy='mean')
binning = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans', subsample=200000)

#classifier = BaggingClassifier()
#classifier = RandomForestClassifier()
#classifier = AdaBoostClassifier(n_estimators = 400)
#classifier = DecisionTreeClassifier()
#classifier = ExtraTreesClassifier()
classifier = HistGradientBoostingClassifier()
#classifier = DecisionTreeClassifier(max_depth=12)
#classifier = GradientBoostingClassifier(n_estimators=300, max_depth=5, subsample=1.0)
#classifier = MLPClassifier(hidden_layer_sizes=(30,30,30,30), max_iter=200)
#classifier = SVC(probability=True)

transformers=[
    ('num', scaler, scaled_features),
    ('cat', encoder, categorical_features)]
# Preprocessing
if impute_flag:
    transformers.append(('imputer', imputer, imputed_featuers))
if binning_flag:
    transformers.append(('bin', binning, binned_features))

pre = ColumnTransformer(transformers=transformers)
pipeline = Pipeline(steps=[('preprocessor', pre), ('classifier', classifier)])

if testing_flag:
    cv_scores = cross_val_score(pipeline, features, labels, cv=3, scoring='roc_auc')
    print('Cross-validation scores')
    print(f'Mean: {cv_scores.mean():.4f} STD: {cv_scores.std():.4f}')
else:
    # Fitting
    pipeline.fit(features, labels)
    # Predicting
    preds = pipeline.predict_proba(testing_data)[:, 1]
    df = pd.DataFrame()
    df['ID'] = testing_data['ID']
    df['income>50k'] = preds
    df.to_csv('preds.csv', index = False)
    print('done')