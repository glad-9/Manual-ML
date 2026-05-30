import yaml

from data_processing.pipeline import pipeline
from core.builder import build_network

def main():
    config_path = "experiments/configs/diabetes.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    dc = config["dataset"]
    subsets = pipeline(dc["path"], dc["label"], dc["train_ratio"], dc["val_ratio"])

    X_train, y_train = subsets["train"]
    X_test, y_test = subsets["test"]
    network = build_network(config_path, X_train, y_train)

    tc = config["training"]
    train_cost, cv_cost = network.fit(
        val_data=subsets["cv"],
        lr=tc["optimizer"]["lr"],
        lambda_reg=tc["lambda_reg"],
        iterations=tc["iterations"],
        save_path="saved_models/diabetes.pkl",
    )

    results = network.evaluate(X_test, y_test)
    print(f"---------------------------\nFinal Costs:\nFinal Train Cost: {train_cost}\nFinal CV Cost: {cv_cost}")
    print(f"---------------------------\nTest Set Results:\nTest Cost: {results["cost"]}\nTest Accuracy: {results["accuracy"]}")


if __name__ == '__main__':
    main()
