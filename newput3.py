import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

def preprocess_data(input_data):
    input_data['date'] = pd.to_datetime(input_data['date'])
    input_data['year'] = input_data['date'].dt.year
    input_data['month'] = input_data['date'].dt.month
    input_data['day'] = input_data['date'].dt.day
    input_data['day_of_week'] = input_data['date'].dt.dayofweek

    label_encoder = LabelEncoder()
    categorical_columns = ['family', 'locale', 'city', 'store_type']
    for col in categorical_columns:
        input_data[col + '_encoded'] = label_encoder.fit_transform(input_data[col])

    input_data.drop(['date'] + categorical_columns, axis=1, inplace=True)

    return input_data

def load_and_predict(date, store_nbr, family, onpromotion, locale, transferred, dcoilwtico, city, store_type, cluster, transactions):
    model = joblib.load("model.pkl") 


    input_data = pd.DataFrame({
        "date": [date],
        "store_nbr": [store_nbr],
        "family": [family],
        "onpromotion": [onpromotion],
        "locale": [locale],
        "transferred": [transferred],
        "dcoilwtico": [dcoilwtico],
        "city": [city],
        "store_type": [store_type],
        "cluster": [cluster],
        "transactions": [transactions]
    })

    input_data = preprocess_data(input_data)

    predictions = model.predict(input_data)

    for prediction in predictions:
        print(f"Sales Prediction: {prediction}")

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description="Sales Forecasting")

   
    parser.add_argument("date", type=str)
    parser.add_argument("store_nbr", type=int)
    parser.add_argument("family", type=str)
    parser.add_argument("onpromotion", type=int)
    parser.add_argument("locale", type=str)
    parser.add_argument("transferred", type=bool)
    parser.add_argument("dcoilwtico", type=float)
    parser.add_argument("city", type=str)
    parser.add_argument("store_type", type=str)
    parser.add_argument("cluster", type=float)
    parser.add_argument("transactions", type=float)


    args = parser.parse_args()

    
    load_and_predict(args.date, args.store_nbr, args.family, args.onpromotion, args.locale, args.transferred, args.dcoilwtico, args.city, args.store_type, args.cluster, args.transactions)
