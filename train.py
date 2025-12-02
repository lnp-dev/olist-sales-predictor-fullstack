import pandas as pd
import joblib
from xgboost import XGBRegressor
import sys
import argparse

DATA_PATH = 'olist_orders_dataset.csv'
ITEMS_PATH = 'olist_order_items_dataset.csv'
MODEL_PATH = 'xgboost_sales_model.pkl'

#load the data
#preprocess data
#create features
#train the model
#execute

def load_data():
    orders = pd.read_csv(DATA_PATH)
    items = pd.read_csv(ITEMS_PATH)

    df = pd.merge(orders, items, on='order_id')
    print('Successfully loaded datasets')
    return df

    
def preprocess_data(df):
    
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df = df.set_index('order_purchase_timestamp').resample("W-MON")['price'].sum().reset_index()
    df = df.rename(columns={'order_purchase_timestamp': 'Week', 'price': 'TotalSales'})
    df = df[df['Week'] >= '2017-01-09']
    df = df.iloc[:-1]
    df = df.reset_index(drop=True)
    print("Successfully preprocessed data")
    return df


def create_features(df):

    #feature 1: lag_1 (Sales from 1 week ago)
    df['sales_last_week'] = df['TotalSales'].shift(1)
    
    #feature 2: lag_4 (Sales from 1 month ago)
    df['sales_1_month_ago'] = df['TotalSales'].shift(4)
    
    #feature 3: Rolling mean (Trend of last 4 weeks)
    df['trend_4w'] = df['TotalSales'].rolling(window=4).mean().shift(1)
    
    #feature 4: week of year
    df['week_of_year'] = df['Week'].dt.isocalendar().week.astype(int)
        
    #Cleaning the dataset
    df = df.dropna()
    return df

def train_model(df, args):

    features = ['sales_last_week', 'sales_1_month_ago', 'trend_4w', 'week_of_year']
    target = 'TotalSales'

    X = df[features]
    y = df[target]

    model = XGBRegressor(
        n_estimators=args.n_estimators, 
        learning_rate=args.learning_rate, 
        random_state=args.random_state
    )
    model.fit(X, y)
    print("Successfully trained XGBoost")
    return model, features

def get_args():
    parser = argparse.ArgumentParser(description="Train XGBoost model for sales forecasting")
    parser.add_argument('--n-estimators', type=int, default=1000, help='Number of trees in the XGBoost model')
    parser.add_argument('--learning-rate', type=float, default=0.05, help='Learning rate for the XGBoost model')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    return parser.parse_args()

if __name__ == '__main__':

    data = load_data()
    data = preprocess_data(data)
    data = create_features(data)
    #The training parameters for XGBoost will be given as command line arguments
    args = get_args()
    model, features = train_model(data, args)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(features, 'model_features.pkl')

    print("Done!")

    
    
