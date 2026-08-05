import matplotlib

matplotlib.use("module://matplotlib-backend-kitty")
import matplotlib.pyplot as plt


def plot_loss(train_history, val_history=None, title="Loss Curve"):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        range(len(train_history)),
        train_history,
        label="Train Loss",
        color="#2563eb",
        linewidth=2,
    )

    if val_history is not None:
        ax.plot(
            range(len(val_history)),
            val_history,
            label="Val Loss",
            color="#dc2626",
            linewidth=2,
            linestyle="--",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()
