#Dependencies
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import joblib
import uvicorn
import pandas as pd
import numpy as np
from datetime import timedelta, datetime as dt
from train import load_data, preprocess_data, create_features, train_model
from pydantic import BaseModel, Field
from typing import Optional
import sys
import subprocess
import requests
import os
import time


app = FastAPI(title="Olist Sales Forecaster")

# --- CORS CONFIGURATION ---
# Configure Cross-Origin Resource Sharing (CORS) to allow the React frontend
# to communicate with this FastAPI backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

df = load_data()
df = preprocess_data(df)
model = joblib.load('xgboost_sales_model.pkl')
features = joblib.load('model_features.pkl')

@app.get('/')
def home():
    return {'status': 'System is healthy.', 'model_version': '1.0'}

@app.get('/predict')
def get_forecast(weeks:int = 12):
    '''
    Takes the number of weeks as input and returns a JSON of forecasted weekly sales 
    for that many weeks after the last known date.
    '''
    #array of Total Sales for most recent 4 weeks
    last_four_weeks = df['TotalSales'].iloc[-4:].tolist()
    #list of predictions for future weeks
    predictions = []
    most_recent_date = df['Week'].iloc[-1]

    for _ in range(weeks):

        #Create features
        lag_1 = last_four_weeks[3]
        lag_4 = last_four_weeks[0]
        trend_4w = np.mean(last_four_weeks)
        most_recent_date += timedelta(weeks=1)
        week_num = most_recent_date.isocalendar().week

        #Predict
        next_features = [lag_1, lag_4, trend_4w, week_num]
        next_features_df = pd.DataFrame([next_features], columns=features)
        prediction = float(model.predict(next_features_df)[0])

        #Update Predictions list and last_four_weeks array
        predictions.append({'date': most_recent_date.strftime('%Y-%m-%d'), 'sales': prediction})
        last_four_weeks.pop(0)
        last_four_weeks.append(prediction)
    
    result = {
        'message': 'Success.',
        'forecast': predictions
    }
    return result

@app.get('/config')
def get_model_params():
    '''Returns the current hyperparameters of the XGBoost model.'''

    params = model.get_params()

    return {
        'message': 'Success.',
        'Hyperparameters': {
            'n_estimators': params['n_estimators'],
            'learning_rate': params['learning_rate'],
            'random_state': params['random_state'],
        } 
    }


class RetrainConfig(BaseModel):
    n_estimators: Optional[int] = None
    learning_rate: Optional[float] = None
    random_state: Optional[int] = None

@app.post('/retrain')
def retrain_model(config: RetrainConfig, background_tasks: BackgroundTasks):
    '''Endpoint to retrain the model with given configuration parameters in the background.'''
    
    background_tasks.add_task(run_training, config)

    return {
        'message': 'Model retraining started in background.',
        'status': 'QUEUED'
    }


def run_training(config: RetrainConfig):
    '''Runs the train.py script with the given configuration parameter and reload application.'''

    args = [sys.executable, 'train.py']
    if config.n_estimators is not None:
        args.extend(['--n-estimators', str(config.n_estimators)])
    if config.learning_rate is not None:
        args.extend(['--learning-rate', str(config.learning_rate)])
    if config.random_state is not None:
        args.extend(['--random-state', str(config.random_state)])
    subprocess.run(args, check=True)

    #Wait to ensure overwrite of Model pkl and reload the model
    time.sleep(2)
    global model
    model = joblib.load('xgboost_sales_model.pkl')

def run_server():
    uvicorn.run("app:app", host='127.0.0.1', port=8000, log_level='info')


if __name__ == '__main__':
    run_server()

