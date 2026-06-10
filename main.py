import yaml

from cli.mnist_mlp import run_mnist_mlp

from data_processing.tabular.tabularpipeline import TabularPipeline
from viz.training import plot_loss
from viz.evaluation import plot_confusion_matrix

from nn.builder import build_network


def main():
    config_path = "experiments/mnist-mlp/mnist-mlp.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # dc = config["dataset"]
    #
    # pipeline = TabularPipeline(
    #     path=dc["path"],
    #     target_column=dc["label"],
    #     categorical_columns=dc["categorical"],
    #     drop_columns=dc["drop"],
    #     train_ratio=dc["train_ratio"],
    #     val_ratio=dc["val_ratio"],
    #     normalize=dc["normalize"],
    # )

    subsets = run_mnist_mlp()

    X_train, y_train = subsets["train"].get_all()
    X_val, y_val = subsets["cv"].get_all()
    X_test, y_test = subsets["test"].get_all()

    feature_count = X_train.shape[1]  # (samples, features)
    model = build_network(config_path, feature_count)

    tc = config["training"]
    train_cost, cv_cost = model.fit(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        iterations=tc["iterations"],
        save_path="saved_models/mnist-mlp.pkl",
    )

    results, y_hat, y_true = model.evaluate(X_test, y_test)
    print(
        f"---------------------------\nFinal Costs:\nFinal Train Cost: {train_cost[-1]}\nFinal Best CV Cost: {model.best_val_loss}"
    )
    print(
        f"---------------------------\nTest Set Results:\nTest Cost: {results['loss']}\nTest Accuracy: {results['accuracy']}"
    )

    plot_loss(train_cost, cv_cost)

    plot_confusion_matrix(y_hat, y_true)


if __name__ == "__main__":
    main()
