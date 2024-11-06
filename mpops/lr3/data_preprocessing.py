import argparse
from typing import List
import pandas as pd


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


def process_data(input_path: str, output_path: str):
    df = load_csv(input_path)
    if df is not None:
        df = drop_columns(df)
        num_columns = collect_numerical_columns(df)
        df_clean = remove_outliers(df, num_columns)
        save_csv(df_clean, output_path)


if __name__ == "__main__":
    args = argparser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    process_data(input_path, output_path)
