import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import metrics

car_dataset = pd.read_csv("E:/Car price Prediction/CAR DETAILS FROM CAR DEKHO.csv")

car_dataset['car_age'] = 2025 - car_dataset['year']
car_dataset.drop(['year'], axis=1, inplace=True)


car_dataset.replace({'fuel': {'Petrol': 0, 'Diesel': 1, 'CNG': 2, 'LPG': 3, 'Electric': 4}}, inplace=True)
car_dataset.replace({'seller_type': {'Dealer': 0, 'Individual': 1, 'Trustmark Dealer': 2}}, inplace=True)
car_dataset.replace({'transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)
car_dataset.replace({'owner': {'First Owner': 0, 'Second Owner': 1, 'Third Owner': 2,
                                'Fourth & Above Owner': 3, 'Test Drive Car': 4}}, inplace=True)

label_encoder = LabelEncoder()
car_dataset['name'] = label_encoder.fit_transform(car_dataset['name'])

joblib.dump(label_encoder, 'label_encoder.pkl')


X = car_dataset.drop(['selling_price'], axis=1)
Y = car_dataset['selling_price']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


rf_model = RandomForestRegressor(n_estimators=100, random_state=2)


rf_model.fit(X_train_scaled, Y_train)

training_data_prediction = rf_model.predict(X_train_scaled)
error_score = metrics.r2_score(Y_train, training_data_prediction)
print(f"R-Squared Error on Training Data: {error_score:.4f}")

joblib.dump(rf_model, 'car_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
