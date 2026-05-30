
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset_builder import *
from train import *
from sklearn.metrics import f1_score


def create_splits(X, y):

    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=False)

    return (
        x_train,x_test,y_train,y_test
    )


def train_lr_workflow(x_train,y_train,x_test,y_test):

    lr_model = create_lr_pipeline()

    lr_model = train_model(lr_model,x_train,y_train)

    evaluate_model(lr_model,x_test,y_test)

    return lr_model


def train_xgb_workflow(x_train,y_train,x_test,y_test):

    scale_pos = (
        y_train.value_counts()[0]/y_train.value_counts()[1]
        )

    xgb_model = create_xgb_pipeline(scale_pos)

    xgb_model = train_model(xgb_model,x_train,y_train)

    evaluate_model(xgb_model,x_test,y_test)

    return xgb_model


def run_training_pipeline(target_ticker,sector_name,sector_dict,save_models=True):

    tickers = prepare_ticker_universe(target_ticker, sector_name, sector_dict)
    raw_data = download_data(tickers)

    data = prepare_dataset(target_ticker,sector_name,sector_dict)

    X , y = data.drop(columns=['Target']),data['Target']

    (x_train,x_test,y_train,y_test) = create_splits(X, y)

    print("Logistic Regression Results")

    lr_model = train_lr_workflow(x_train,y_train,x_test,y_test)

    print("\nXGBoost Results")

    xgb_model = train_xgb_workflow(x_train,y_train,x_test,y_test)
    f1_score_lr = f1_score(y_test,lr_model.predict(x_test))
    f1_score_xgb = f1_score(y_test,xgb_model.predict(x_test))


    lr_weight = f1_score_lr/(f1_score_lr+f1_score_xgb)
    xgb_weight = f1_score_xgb/(f1_score_lr+f1_score_xgb)

    if save_models:
        save_model(lr_model, "logistic_regression.pkl")
        save_model(xgb_model, "xgboost.pkl")
        print("\nModels saved.")

    return {
        'raw_data':raw_data,
        'lr_weight':lr_weight,
        'xgb_weight':xgb_weight,
        "main_feature_matrix": X,
        "lr_model": lr_model,
        "xgb_model": xgb_model,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test
        }