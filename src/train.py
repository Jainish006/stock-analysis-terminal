
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from xgboost import XGBClassifier



def create_lr_pipeline():

    pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),

            (
                "lr_model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000
                )
            )
        ]
    )

    return pipeline


def create_xgb_pipeline(scale_pos_weight):

    pipeline = Pipeline(
        steps=[
            (
                "xgb_model",
                XGBClassifier(
                    max_depth=5,
                    learning_rate=0.008,
                    n_estimators=10,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42
                )
            )
        ]
)

    return pipeline



def train_model(model,x_train,y_train):

    model.fit(x_train,y_train)

    return model



def predict(model,X):

    return model.predict(X)


def predict_proba(model,X):

    return model.predict_proba(X)




def evaluate_model(model,x_test,y_test):

    predictions = model.predict(x_test)

    print(classification_report(y_test,predictions))



def save_model(model,file_path):

    joblib.dump(model,file_path)


def load_model(file_path):
    return joblib.load(file_path)