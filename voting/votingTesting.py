import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

# Importing data
training_data = pd.read_csv("../ml-2024-f/train_final.csv")
testing_data = pd.read_csv("../ml-2024-f/test_final.csv")

# Splitting into features and labels
features = training_data.drop("income>50K", axis=1)
labels = training_data["income>50K"]

# Corrected column names
numeric_features = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
categorical_features = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

scaler = StandardScaler()
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
#encoder = OrdinalEncoder()

# Create classifier instances
gb_classifier = GradientBoostingClassifier(n_estimators=100)
dt_classifier = DecisionTreeClassifier(max_depth=2)
b_classifier = BaggingClassifier(max_samples=8)
rf_classifier = RandomForestClassifier(max_depth=6)
et_classifier = ExtraTreesClassifier(max_depth=4)
mlp_classifier = MLPClassifier(hidden_layer_sizes=(30,15), max_iter=400)

# Combine classifiers into a VotingClassifier
voting_classifier = VotingClassifier(estimators=[('gradient_boosting', gb_classifier), ('random_forest', rf_classifier), ('mlp', mlp_classifier)], voting='soft')

# Preprocessing
pre = ColumnTransformer(transformers=[('num', scaler, numeric_features), ('cat', encoder, categorical_features)])
pipeline = Pipeline(steps=[('preprocessor', pre), ('classifier', voting_classifier)])

f_train, f_test, l_train, l_test = train_test_split(features, labels, test_size=.25)

# Fitting
pipeline.fit(f_train, l_train)

# Predicting
preds_testing = pipeline.predict(f_test)
preds_training = pipeline.predict(f_train)

accuracy_testing = accuracy_score(l_test, preds_testing)
accuracy_training = accuracy_score(l_train, preds_training)
print(f'Testing Accuracy: {accuracy_testing:.2f}')
print(f'Training Accuracy: {accuracy_training:.2f}')
