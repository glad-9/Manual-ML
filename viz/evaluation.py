import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_hat, y_true, labels=None, title="Confusion Matrix"):

    if hasattr(y_true, "data"):
        y_true = y_true.copy()  # y_true for some reason is a np array
    if hasattr(y_hat, "data"):
        y_hat = y_hat.data.copy().get()

    y_true = y_true.flatten().astype(int)
    y_hat = y_hat.flatten().astype(int)

    classes = np.unique(np.concatenate([y_true, y_hat]))
    n = len(classes)

    if labels is None:
        labels = [str(c) for c in classes]

    matrix = np.zeros((n, n), dtype=int)
    for true, pred in zip(y_true, y_hat):
        matrix[true][pred] += 1

    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(matrix, cmap="Blues")

    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = matrix.max() / 2
    for i in range(n):
        for j in range(n):
            color = "white" if matrix[i, j] > thresh else "black"
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color=color,
                fontsize=12,
            )

    plt.tight_layout()
    plt.show()
