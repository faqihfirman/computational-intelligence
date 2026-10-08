import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


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

        display_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        ax.imshow(display_image)
        ax.axis("off")
        ax.set_title(f"True: {true_label}\nPred: {pred_label} ({confidence:.1%})",
                     color=color, fontsize=10)

    plt.tight_layout()
    plt.show()


def make_gradcam_heatmap(image, model, last_conv_layer_name, pred_index=None):
    layer_names = [layer.name for layer in model.layers]
    if last_conv_layer_name not in layer_names:
        conv_names = [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
        if not conv_names:
            raise ValueError(f"No Conv2D layer found in model. Layers: {layer_names}")
        last_conv_layer_name = conv_names[-1]

    inputs = tf.keras.Input(shape=image.shape)
    x = inputs
    conv_tensor = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_tensor = x
    grad_model = tf.keras.models.Model(inputs, [conv_tensor, x])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image[np.newaxis, ...])
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)


def overlay_gradcam(image, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    base_image = image
    if base_image.ndim == 2 or base_image.shape[-1] == 1:
        base_image = cv2.cvtColor(np.squeeze(base_image), cv2.COLOR_GRAY2RGB)
    base_image_uint8 = np.uint8(255 * base_image) if base_image.max() <= 1.0 else base_image.astype(np.uint8)

    overlay = cv2.addWeighted(base_image_uint8, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


def plot_gradcam_samples(model, X_samples, y_true_idx, label_encoder, last_conv_layer_name, n_samples=10, seed=42):
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(X_samples), size=n_samples, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(15, 7))

    for ax, idx in zip(axes.flatten(), sample_indices):
        image = X_samples[idx]
        true_label = label_encoder.inverse_transform([y_true_idx[idx]])[0]

        heatmap, pred_idx = make_gradcam_heatmap(image, model, last_conv_layer_name)
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        overlay = overlay_gradcam(image, heatmap)

        is_correct = pred_label == true_label
        color = "green" if is_correct else "red"

        ax.imshow(overlay)
        ax.axis("off")
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=10)

    plt.tight_layout()
    plt.show()
