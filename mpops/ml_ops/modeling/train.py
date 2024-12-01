import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, roc_curve
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
import mlflow
from ml_ops.create_bucket import create_bucket
from ml_ops.data_upload import upload_file
from joblib import dump
import os
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt


argparser = argparse.ArgumentParser()
argparser.add_argument("-p", "--params", required=True, help="File path to params")
argparser.add_argument("-d", "--data_path", required=True, help="File path to data")

mlflow.set_tracking_uri("http://localhost:5000")
try:
    mlflow.create_experiment("Experiment", artifact_location="s3://mlflow")
except mlflow.MlflowException as e:
    print(e)
mlflow.set_experiment("Experiment")


def load_data(file_path: str):
    if not file_path.endswith(".csv"):
        raise ValueError(f"Wrong file type: {file_path}. Expected .csv")
    try:
        df = pd.read_csv(file_path, delimiter=",")
        X, y = df.drop(columns=['Churn']), df['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.3,
                                                            random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise
    except Exception as e:
        print(f"Error while reading file: {e}")
        raise


def train_model_SVC(X_train, X_test, y_train, y_test, params):
    with mlflow.start_run(run_name="Support vector machine"):
        _C_range_rbf = params[0]['C_rbf']['range']
        C_range_rbf = loguniform(_C_range_rbf[0], _C_range_rbf[1])  # диапазон для гиперпараметра регуляризации, будем использовать для ядра rbf
        _gamma_range = params[0]['gamma']['range']
        gamma_range = loguniform(_gamma_range[0], _gamma_range[1])  # диапазон для гиперпараметра коэффициента ядра (применим к ядрам rbf, poly, sigmoid)
        _C_range_poly = params[1]['C_poly']['range']
        C_range_poly = loguniform(_C_range_poly[0], _C_range_poly[1])  # диапазон для гиперпараметра регуляризации, будем использовать для ядра poly
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': gamma_range,
                            'C': C_range_rbf, },
                            {'kernel': ['poly'], 'degree': params[1]['degree'], 'C': C_range_poly, }]  # degree - Степень полиномиальной функции ядра (‘poly’)
        SVC_search = RandomizedSearchCV(estimator=SVC(coef0=0.5, probability=True), verbose=3,
                                        param_distributions=tuned_parameters,
                                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=27),
                                        n_iter=20)
        SVC_search.fit(X_train, y_train)
        SVCbest = SVC_search.best_params_
        SVCbest["probability"] = True
        svc = SVC(**SVCbest).fit(X_train, y_train)

        mlflow.log_param("gamma", SVCbest.get('gamma'))
        mlflow.log_param("kernel", SVCbest.get('kernel'))
        mlflow.log_param("C", SVCbest.get('C'))

        metrics = [f1_score, accuracy_score, precision_score, recall_score]
        metric_names = ['f1_score', 'accuracy_score', 'precision_score', 'recall_score']

        for name, metric, in zip(metric_names, metrics):
            _metric_train = calculate_metric(svc, X_train, y_train, metric)
            _metric_val = calculate_metric(svc, X_test, y_test, metric)
            print(name + f" на тренировочной выборке: {_metric_train: .4f}")
            print(name + f" на валидационной выборке: {_metric_val: .4f}")
            mlflow.log_metric(name, _metric_train)
            mlflow.log_metric(name, _metric_val)

        y_pred = svc.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=np.array(['no', 'yes']), output_dict=True)
        report_path = "reports/classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f)

        create_bucket("mlflow")
        upload_file("mlflow", f'artifacts/{mlflow.active_run().info.run_id}/classification_report.json', report_path)

        fprate, tprate, _ = roc_curve(y_test, svc.predict_proba(X_test)[:, 1])
        plt.figure()
        plt.plot(fprate, tprate, color='g')
        plt.plot([0, 1], [0, 1], color='r', linestyle='--')
        plt.title("ROC curve")
        plt.ylabel("true positive rate, TPR")
        plt.xlabel("false positive rate, FPR")
        plt.grid(color='w')
        figure_path = "reports/figures/roc_curve.png"
        plt.savefig(figure_path)

        upload_file("mlflow", f'artifacts/{mlflow.active_run().info.run_id}/roc_curve.png', figure_path)

        model_name = (f'SVC_'
                      f'{SVCbest["gamma"]}_'
                      f'{SVCbest["kernel"]}_'
                      f'{SVCbest["C"]}'
                      f'.joblib')

        dump(SVC_search.best_estimator_, os.path.join("models", model_name))
        create_bucket("model")
        upload_file("model", f'experiments/{mlflow.active_run().info.run_id}/{model_name}', f"models/{model_name}")


def calculate_metric(model_pipe, X, y, metric=f1_score):
    y_model = model_pipe.predict(X)
    return metric(y, y_model)


def prepare_params(**kwargs):
    params = {
        k0: {
            k1: kwargs.get(k0, {}).get(k1, v1) for k1, v1 in v0.items()
        } if type(v0) in {dict, DictConfig}
        else kwargs.get(k0, v0)
        for k0, v0 in kwargs.items()
    }
    return params


if __name__ == "__main__":
    args = argparser.parse_args()

    X_train, X_val, y_train, y_val = load_data(args.data_path)
    params = prepare_params(**OmegaConf.load(args.params))

    mlflow.set_experiment("Experiment")

    train_model_SVC(X_train, X_val, y_train, y_val, params['random_search'])
