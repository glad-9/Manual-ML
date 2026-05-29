import pandas as pd

def normalize(train_df, other_dfs, target_column=None):

    if other_dfs is None:
        other_dfs = []

    train_df = train_df.copy()
    numeric_columns = train_df.select_dtypes(include=['int64','float64']).columns

    if target_column:
        numeric_columns = numeric_columns.drop(target_column, errors='ignore')

    means = {}
    stds = {}

    for col in numeric_columns:
        means[col] = train_df[col].mean()
        stds[col] = train_df[col].std()

        if stds[col] == 0:
            stds[col] = 1

        train_df[col] = (train_df[col] - means[col]) / stds[col]

    normalized_others = []
    for df in other_dfs:
        df = df.copy()

        for col in numeric_columns:
            df[col] = (df[col] - means[col]) / stds[col]

        normalized_others.append(df)

    return train_df, normalized_others

