import yaml

from cli.args import parse_args

from data_processing.tabular.tabularpipeline import TabularPipeline
from data_processing.image.imagepipeline import ImagePipeline

from viz.training import plot_loss
from viz.evaluation import plot_confusion_matrix

from nn.builder import build_network


def main():
    args = parse_args()
    assert args is not None

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dc = config["dataset"]

    if dc["modality"] == "tabular":
        if args.dataset:
            dc["path"] = args.dataset

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

    elif dc["modality"] == "image":
        if args.dataset:
            dc["images_path"] = args.dataset
        if args.labels:
            dc["labels_path"] = args.labels

        pipeline = ImagePipeline(
            images_path=dc["images_path"],
            labels_path=dc["labels_path"],
            train_ratio=dc["train_ratio"],
            val_ratio=dc["val_ratio"],
            normalize=dc["normalize"],
        )
        subsets = pipeline.run(idx=True, one_hot=True)

    X_train, y_train = subsets["train"].get_all()
    X_val, y_val = subsets["cv"].get_all()
    X_test, y_test = subsets["test"].get_all()

    feature_count = X_train.shape[1]  # (samples, features)
    model = build_network(args.config, feature_count)

    tc = config["training"]
    save_path = args.save_path

    train_cost, cv_cost = model.fit(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        max_epochs=tc["max_epochs"],
        patience=10,
        target_val_loss=0.05,
        save_path=save_path,
    )

    results, y_hat, y_true = model.evaluate(X_test, y_test)
    print(
        f"---------------------------\nFinal Loss:\nFinal Train Loss: {train_cost[-1]}\nFinal Best CV Cost: {model.best_val_loss}"
    )
    print(
        f"---------------------------\nTest Set Results:\nTest Loss: {results['loss']}\nTest Accuracy: {results['accuracy']}"
    )

    plot_loss(train_cost, cv_cost)

    # plot_confusion_matrix(y_hat, y_true)


if __name__ == "__main__":
    main()
