import yaml

from data_processing.tabular.tabularpipeline import TabularPipeline
from core.tensor import Tensor
from nn.builder import build_network

def main():
    config_path = "experiments/configs/diabetes.yaml"

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
    network = build_network(config_path, feature_count)

    X_train = Tensor(X_train)
    y_train = Tensor(y_train.reshape(-1,1))
    X_val = Tensor(X_val)
    y_val = Tensor(y_val.reshape(-1,1))
    X_test = Tensor(X_test)
    y_test = Tensor(y_test.reshape(-1,1))

    tc = config["training"]
    train_cost, cv_cost = network.fit(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        iterations=tc["iterations"],
        save_path="saved_models/diabetes.pkl",
    )

    results = network.evaluate(X_test, y_test)
    print(f"---------------------------\nFinal Costs:\nFinal Train Cost: {train_cost}\nFinal CV Cost: {cv_cost}")
    print(f"---------------------------\nTest Set Results:\nTest Cost: {results["loss"]}\nTest Accuracy: {results["accuracy"]}")


if __name__ == '__main__':
    main()
