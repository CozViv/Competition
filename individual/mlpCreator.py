import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
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
# Change this to use more encoders in future, on different ones in which ordinal or one hot is better
encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
classifier = MLPClassifier(hidden_layer_sizes=(30,15), max_iter=400)

# Preprocessing
pre = ColumnTransformer(transformers=[('num', scaler, numeric_features), ('cat', encoder, categorical_features)])
pipeline = Pipeline(steps=[('preprocessor', pre), ('classifier', classifier)])

# Fitting
pipeline.fit(features, labels)

# Predicting
preds = pipeline.predict(testing_data)
df = pd.DataFrame()
df['ID'] = testing_data['ID']
df['income>50k'] = preds
df.to_csv('mlpPreds.csv', index = False)