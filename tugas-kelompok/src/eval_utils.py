import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(history, metric="f1_score", metric_label="F1 Score (macro)"):
    fig, (metric_ax, loss_ax) = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs_range = range(1, len(history.history["loss"]) + 1)

    metric_ax.plot(epochs_range, history.history[metric], label="Train", color="navy")
    metric_ax.plot(epochs_range, history.history[f"val_{metric}"], label="Val", color="crimson")
    metric_ax.set_title(f"{metric_label} per Epoch")
    metric_ax.set_xlabel("Epoch")
    metric_ax.set_ylabel(metric_label)
    metric_ax.legend()
    metric_ax.grid(alpha=0.3)

    loss_ax.plot(epochs_range, history.history["loss"], label="Train", color="navy")
    loss_ax.plot(epochs_range, history.history["val_loss"], label="Val", color="crimson")
    loss_ax.set_title("Loss per Epoch")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.legend()
    loss_ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_sample_predictions(model, paths_test, y_test, label_encoder, predict_fn, n_samples=10, seed=42):
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(paths_test), size=n_samples, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(15, 7))

    for ax, idx in zip(axes.flatten(), sample_indices):
        image_path = paths_test[idx]
        true_label = label_encoder.inverse_transform([np.argmax(y_test[idx])])[0]

        pred_label, confidence, image = predict_fn(model, image_path)
        is_correct = pred_label == true_label
        color = "green" if is_correct else "red"

        ax.imshow(image, cmap="gray")
        ax.axis("off")
        ax.set_title(f"True: {true_label}\nPred: {pred_label} ({confidence:.1%})",
                     color=color, fontsize=10)

    plt.tight_layout()
    plt.show()
