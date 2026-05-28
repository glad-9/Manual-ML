from .loaders import load_csv
from .train_test_split import train_test_split
from .preprocessing import normalize
from .features import split_features_labels, reshape_inputs

def pipeline(csv_path, label):
    # Load raw dataset
    raw_df = load_csv(csv_path)


    # Split dataset
    train_df, val_df, test_df = train_test_split(raw_df)


    # Normalize
    train_df, [val_df, test_df] = normalize(
        train_df,
        [val_df, test_df]
    )
    
    # Convert to features/labels
    X_train, y_train = split_features_labels(train_df, label)
    X_val, y_val = split_features_labels(val_df, label)
    X_test, y_test = split_features_labels(test_df, label)


    # Convert to neural-network shapes
    X_train, y_train = reshape_inputs(X_train, y_train)
    X_val, y_val = reshape_inputs(X_val, y_val)
    X_test, y_test = reshape_inputs(X_test, y_test)

    return {"train": (X_train, y_train),
            "cv": (X_val, y_val),
            "test": (X_test, y_test)}