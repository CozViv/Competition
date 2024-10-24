import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

# Preprocessing
pre = ColumnTransformer(transformers=[('num', scaler, numeric_features), ('cat', encoder, categorical_features)])
pipeline = Pipeline(steps=[('preprocessor', pre), ('classifier', HistGradientBoostingClassifier())])

#f_train, f_test, l_train, l_test = train_test_split(features, labels, test_size=.01, random_state=45)
f_train = features
l_train = labels

# Fitting
pipeline.fit(f_train, l_train)
# Predicting
preds = pipeline.predict(testing_data)
df = pd.DataFrame()
df['ID'] = testing_data['ID']
df['income>50k'] = preds
df.to_csv('histGradientBoostingPreds.csv', index = False)