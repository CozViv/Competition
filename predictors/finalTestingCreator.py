import pandas as pd
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
binning_flag = True

# Importing data
training_data = pd.read_csv("../ml-2024-f/train_final.csv")
testing_data = pd.read_csv("../ml-2024-f/test_final.csv")

# Dropping duplicate data
training_data.drop("education", axis=1)
testing_data.drop("education", axis=1)

# Dropping poor features
# training_data.drop("race", axis=1)
# testing_data.drop("race", axis=1)
# training_data.drop("workclass", axis=1)
# testing_data.drop ("workclass", axis=1)

# Splitting into features and labels
features = training_data.drop("income>50K", axis=1)
labels = training_data["income>50K"]

scaled_features = []
imputed_featuers = ['capital.gain', 'capital.loss']
categorical_features = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
binned_features = ['age','fnlwgt', 'hours.per.week', 'capital.gain', 'capital.loss']

# training_data.drop("native.country", axis=1)
# testing_data.drop("native.country", axis=1)
# training_data.drop("sex", axis=1)
# testing_data.drop("sex", axis=1)
# training_data.drop("race", axis=1)
# testing_data.drop("race", axis=1)
# training_data.drop("capital.gain", axis=1)
# testing_data.drop ("capital.gain", axis=1)
# training_data.drop("capital.loss", axis=1)
# testing_data.drop ("capital.loss", axis=1)

# numeric_features = ['age', 'fnlwgt', 'education.num', 'hours.per.week']
# scaled_features = ['fnlwgt', 'hours.per.week']
# imputed_featuers = ['capital.gain', 'capital.loss']
# categorical_features = ['workclass', 'marital.status', 'occupation', 'relationship']
# binned_features = ['age']

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
#classifier = HistGradientBoostingClassifier()
#classifier = DecisionTreeClassifier(max_depth=12)
classifier = GradientBoostingClassifier(n_estimators=600)
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
    # f_train, f_test, l_train, l_test = train_test_split(features, labels, test_size=.25)

    # # Fitting
    # pipeline.fit(f_train, l_train)
    # # Predicting
    # preds_testing_probs = pipeline.predict_proba(f_test)[:, 1]
    # preds_training_probs = pipeline.predict_proba(f_train)[:, 1]
    # auc_testing = roc_auc_score(l_test, preds_testing_probs)
    # auc_training = roc_auc_score(l_train, preds_training_probs)
    # print(f'AUC testing: {auc_testing:.4f}')
    # print(f'AUC training: {auc_training:.4f}')
    cv_scores = cross_val_score(pipeline, features, labels, cv=5, scoring='roc_auc')
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