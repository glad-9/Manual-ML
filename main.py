import yaml

from data_processing.tabular.tabularpipeline import TabularPipeline
from core.tensor import Tensor
from nn.builder import build_network

def main():
    config_path = "experiments/configs/housing_prices.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    dc = config["dataset"]
    
    pipeline = TabularPipeline(
        path=dc["path"],
        target_column=dc["label"],
        categorical_columns=dc["categorical"],
        drop_columns=dc["drop"],
        train_ratio=dc["train_ratio"],
        val_ratio=dc["val_ratio"],
        normalize=dc["normalize"],
        )

    subsets = pipeline.run()

    X_train, y_train = subsets["train"].get_all()
    X_val, y_val = subsets["cv"].get_all()
    X_test, y_test = subsets["test"].get_all()

    feature_count = X_train.shape[1] # (samples, features)
    model = build_network(config_path, feature_count)

    tc = config["training"]
    train_cost, cv_cost = model.fit(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        iterations=tc["iterations"],
        save_path="saved_models/diabetes.pkl",
    )

    results = model.evaluate(X_test, y_test)
    print(f"---------------------------\nFinal Costs:\nFinal Train Cost: {train_cost[-1]}\nFinal Best CV Cost: {model.best_val_loss}")
    print(f"---------------------------\nTest Set Results:\nTest Cost: {results["loss"]}\nTest Accuracy: {results["accuracy"]}")


if __name__ == '__main__':
    main()
