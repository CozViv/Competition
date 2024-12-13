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
import xgboost as xgb

impute_flag = True
binning_flag = True

# Importing data
training_data = pd.read_csv("../ml-2024-f/train_final.csv")
testing_data = pd.read_csv("../ml-2024-f/test_final.csv")

# Dropping duplicate data
training_data.drop("education", axis=1)
testing_data.drop("education", axis=1)

def feature_engineering(data):
    data['age_hours'] = data['age'] * data['hours.per.week']
    data['net_capital'] = data['capital.gain'] - data['capital.loss']

    data['log_capital_gain'] = np.log1p(data['capital.gain'])
    data['log_capital_loss'] = np.log1p(data['capital.loss'])

    data['age_group'] = pd.cut(
        data['age'], bins=[0, 25, 45, 65, 100], labels=['young', 'middle-aged', 'senior', 'retired']
    )

    rare_threshold = 100
    native_country_counts = data['native.country'].value_counts()
    data['native.country'] = data['native.country'].apply(
        lambda x: x if native_country_counts[x] > rare_threshold else 'Other'
    )

    age_group_order = {'young': 1, 'middle-aged': 2, 'senior': 3, 'retired': 4}
    data['age_group'] = data['age_group'].map(age_group_order)
    
    return data

# Apply feature engineering to both training and testing data
training_data = feature_engineering(training_data)
testing_data = feature_engineering(testing_data)

# Splitting into features and labels
features = training_data.drop("income>50K", axis=1)
labels = training_data["income>50K"]

scaled_features = ['fnlwgt', 'age','fnlwgt', 'hours.per.week', 'net_capital', 'log_capital_gain']
imputed_featuers = ['capital.gain', 'capital.loss', 'net_capital']
categorical_features = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
binned_features = ['age','fnlwgt', 'hours.per.week']

scaler = StandardScaler()
encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
#imputer = KNNImputer(missing_values=0)
imputer = SimpleImputer(missing_values=0, strategy='mean')
binning = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans', subsample=10000)

#classifier = HistGradientBoostingClassifier(learning_rate=.01, max_iter=1000)
#classifier = HistGradientBoostingClassifier()
classifier = AdaBoostClassifier()
#classifier = RandomForestClassifier()
#classifier = DecisionTreeClassifier()
#classifier = xgb.XGBClassifier(tree_method="hist")
#classifier = GradientBoostingClassifier(n_estimators=300, max_depth=5, subsample=1.0)

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

training_data.drop('age', axis=1)

cv_scores = cross_val_score(pipeline, features, labels, cv=3, scoring='roc_auc')
print('Cross-validation scores')
print(f'Mean: {cv_scores.mean():.4f} STD: {cv_scores.std():.4f}')
# Fitting
pipeline.fit(features, labels)
# Predicting
preds = pipeline.predict_proba(testing_data)[:, 1]
df = pd.DataFrame()
df['ID'] = testing_data['ID']
df['income>50k'] = preds
df.to_csv('preds.csv', index = False)
print('done')