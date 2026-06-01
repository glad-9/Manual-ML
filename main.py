import yaml

from data_processing.tabular.tabularpipeline import TabularPipeline
from data_processing.tabular.normalizer import Normalizer

from core.builder import build_network

def main():
    config_path = "experiments/configs/diabetes.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    dc = config["dataset"]
    
    pipeline = TabularPipeline(dc["path"], [Normalizer()], dc["train_ratio"], dc["val_ratio"])

    subsets = pipeline.run(dc["label"])

    X_train, y_train = subsets["train"]
    X_test, y_test = subsets["test"]

    feature_count = X_train.shape[1] # (samples, features)
    network = build_network(config_path, feature_count)

    tc = config["training"]
    train_cost, cv_cost = network.fit(
        train_data=subsets["train"],
        val_data=subsets["cv"],
        iterations=tc["iterations"],
        save_path="saved_models/diabetes.pkl",
    )

    results = network.evaluate(X_test, y_test)
    print(f"---------------------------\nFinal Costs:\nFinal Train Cost: {train_cost}\nFinal CV Cost: {cv_cost}")
    print(f"---------------------------\nTest Set Results:\nTest Cost: {results["cost"]}\nTest Accuracy: {results["accuracy"]}")


if __name__ == '__main__':
    main()
