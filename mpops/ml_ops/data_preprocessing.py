import argparse
from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


argparser = argparse.ArgumentParser()
argparser.add_argument("-i", "--input_path", required=True, help="Path of the input file")
argparser.add_argument(
    "-o", "--output_path", required=True, help="Path to save the processed data"
)


def load_csv(file_path: str) -> pd.DataFrame:
    if not file_path.endswith(".csv"):
        raise ValueError(f"Wrong file type: {file_path}. Expected .csv")
    try:
        df = pd.read_csv(file_path, delimiter=",")
        df.columns = df.columns.str.replace('_', ' ').str.title()
        df.columns = df.columns.str.replace(' ', '')
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise
    except Exception as e:
        print(f"Error while reading file: {e}")
        raise


def save_csv(df: pd.DataFrame, file_path: str):
    try:
        with open(f"{file_path}", 'w', encoding='utf-8') as f:
            df.to_csv(f, index=False)
    except Exception as e:
        print(f"Error while writing file: {e}")
        raise


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Удалим мусорные признаки"""
    question_vmail = df[df.NumberVmailMessages == 0]
    df.drop(question_vmail.index).reset_index(drop=True)
    df.drop([
        'TotalDayMinutes',
        'TotalEveMinutes',
        'TotalNightMinutes',
        'TotalIntlMinutes'
    ], axis=1).reset_index(drop=True)
    return df


def collect_numerical_columns(df: pd.DataFrame) -> List[str]:
    num_columns = [num for num in df.columns if (df[num].dtypes != object)]
    return num_columns


def remove_outliers(data, labels):
    for label in labels:
        q1 = data[label].quantile(0.25)
        q3 = data[label].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr
        data[label] = data[label].mask(
            data[label] < lower_bound, data[label].median(), axis=0
        )
        data[label] = data[label].mask(
            data[label] > upper_bound, data[label].median(), axis=0
        )

    return data


def preprocess_data_sklearn(df: pd.DataFrame) -> pd.DataFrame:
    num_pipe_standard = Pipeline([
        ('scaler', StandardScaler())
    ])
    num_standard = [
        'AccountLength',
        'TotalDayCalls',
        'TotalDayCharge',
        'TotalEveCalls',
        'TotalEveCharge',
        'TotalNightCalls',
        'TotalNightCharge',
        'TotalIntlCalls',
        'TotalIntlCharge'
    ]

    num_pipe_norm = Pipeline([
        ('norm', MinMaxScaler())
    ])
    num_norm = ['NumberVmailMessages', 'TotalIntlCalls', 'NumberCustomerServiceCalls']

    cat_pipe_ohe = Pipeline([
        ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
    ])
    cat_style_ohe = ['AreaCode']

    cat_pipe_ord = Pipeline([
        ('encoder', OrdinalEncoder())
    ])
    cat_ord = ['State', 'InternationalPlan', 'VoiceMailPlan', 'Churn']

    preprocessors = ColumnTransformer(transformers=[
        ('num_standard', num_pipe_standard, num_standard),
        ('num_norm', num_pipe_norm, num_norm),
        ('cat_style_ohe', cat_pipe_ohe, cat_style_ohe),
        ('cat_ord', cat_pipe_ord, cat_ord),
    ])

    preprocessors.fit(df)
    cat_ohe_names = preprocessors.transformers_[2][1]['encoder'].get_feature_names_out(cat_style_ohe)

    columns = np.hstack([num_standard,
                        num_norm,
                        cat_ohe_names,
                        cat_ord])
    df_transformed = preprocessors.transform(df)
    df_transformed = pd.DataFrame(df_transformed, columns=columns)
    return df_transformed


def process_data(input_path: str, output_path: str):
    df = load_csv(input_path)
    if df is not None:
        df = drop_columns(df)
        num_columns = collect_numerical_columns(df)
        df_clean = remove_outliers(df, num_columns)
        df_transformed = preprocess_data_sklearn(df_clean)
        save_csv(df_transformed, output_path)


if __name__ == "__main__":
    args = argparser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    process_data(input_path, output_path)
