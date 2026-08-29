import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="NumTorch training entry point")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML network configuration file",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override the dataset path from the configuration file",
    )

    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Override the labels path from the configuration file",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Override where the best model checkpoint is saved",
    )

    return parser.parse_args()
