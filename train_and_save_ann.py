import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Load the dataset
df = pd.read_csv("Churn_Modelling.csv")

# Feature selection
X = df[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
y = df['Exited']

# Encode Gender
label_encoder_gender = LabelEncoder()
X['Gender'] = label_encoder_gender.fit_transform(X['Gender'])

# Encode Geography (OneHot)
onehot_encoder_geo = OneHotEncoder(sparse_output=False, drop='first')
geo_encoded = onehot_encoder_geo.fit_transform(X[['Geography']])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine all features
X = X.drop(columns=['Geography']).reset_index(drop=True)
X = pd.concat([X, geo_encoded_df], axis=1)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ANN model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(6, activation='relu'),
    Dense(6, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.1, verbose=1)

# Save everything
model.save("model_new.keras")
with open("label_encoder_gender.pkl", "wb") as f:
    pickle.dump(label_encoder_gender, f)
with open("onehot_encoder_geo.pkl", "wb") as f:
    pickle.dump(onehot_encoder_geo, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Training complete. Model and encoders saved.")
