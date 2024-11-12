import time
from typing import Optional, Callable

import numpy as np
import numpy.typing as npt
import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.training.train_state
import optax
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report, accuracy_score
from tqdm import tqdm

TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'

class TrainState(flax.training.train_state.TrainState):
    # See https://flax.readthedocs.io/en/latest/guides/dropout.html.
    key: jax.random.KeyArray  # type: ignore

@jax.jit
def train_step(state: TrainState, inputs_spectrogram: jax.Array, inputs_tabular: jax.Array, labels: jax.Array, key: jax.random.KeyArray) -> TrainState:
    """
    Performs a single training step on the given batch of inputs and labels.

    Args:
        state: The current training state.
        inputs_spectrogram: Batch of spectrogram inputs.
        inputs_tabular: Batch of tabular inputs.
        labels: The batch of labels.
        key: The random key to use.

    Returns:
        The updated training state.
    """
    key, dropout_key = jax.random.split(key=key)
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            x_spectrogram=inputs_spectrogram,
            x_tabular=inputs_tabular,
            train=True,
            rngs={'dropout': dropout_train_key}
        )
        if logits.shape[-1] <= 2:
            if logits.shape[-1] == 2:
                logits = logits[:, 1]
            loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels).mean()
        else:
            loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()
        return loss

    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state

@jax.jit
def eval_step(state: TrainState, inputs_spectrogram: jax.Array, inputs_tabular: jax.Array, labels: jax.Array) -> tuple[float, jax.Array]:
    """
    Performs a single evaluation step on the given batch of inputs and labels.

    Args:
        state: The current training state.
        inputs_spectrogram: Batch of spectrogram inputs.
        inputs_tabular: Batch of tabular inputs.
        labels: The batch of labels.

    Returns:
        loss: The loss on the given batch.
        logits: The logits on the given batch.
    """
    logits = state.apply_fn(
        {'params': state.params},
        x_spectrogram=inputs_spectrogram,
        x_tabular=inputs_tabular,
        train=False,
        rngs={'dropout': state.key}
    )
    if logits.shape[-1] <= 2:
        if logits.shape[-1] == 2:
            logits = logits[:, 1]
        loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels).mean()
    else:
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()
    return loss, logits

def evaluate(state: TrainState, eval_dataloader, num_classes: int,
             tqdm_desc: Optional[str] = None, debug: bool = False) -> tuple[float, float, float, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
    """
    Evaluates the model given the current training state on the given dataloader.

    Args:
        state: The current training state.
        eval_dataloader: The dataloader to evaluate on.
        num_classes: The number of classes.
        tqdm_desc: The description to use for the tqdm progress bar. If None, no progress bar is shown.
        debug: Whether to print extra information for debugging.

    Returns:
        eval_loss: The average loss.
        eval_accuracy: The accuracy.
        eval_auc: The AUC score.
        y_true: The true labels.
        y_pred_labels: The predicted labels.
        eval_fpr: False positive rates (for ROC curve).
        eval_tpr: True positive rates (for ROC curve).
    """
    logits_list, labels_list = [], []
    eval_loss = 0.0
    with tqdm(total=len(eval_dataloader), desc=tqdm_desc, unit="batch", bar_format=TQDM_BAR_FORMAT, disable=tqdm_desc is None) as progress_bar:
        for inputs_spectrogram_batch, inputs_tabular_batch, labels_batch in eval_dataloader:
            loss_batch, logits_batch = eval_step(state, inputs_spectrogram_batch, inputs_tabular_batch, labels_batch)
            logits_list.append(logits_batch)
            labels_list.append(labels_batch)
            eval_loss += loss_batch
            progress_bar.update(1)
        eval_loss /= len(eval_dataloader)
        logits = jnp.concatenate(logits_list)
        y_true = jnp.concatenate(labels_list)

        # Convert to NumPy arrays for sklearn compatibility
        y_probs = jax.nn.sigmoid(logits) if num_classes == 2 else jax.nn.softmax(logits, axis=-1)
        y_probs_np = np.array(y_probs)
        y_true_np = np.array(y_true)

        # Determine optimal threshold and compute metrics
        if num_classes == 2:
            fpr, tpr, thresholds = roc_curve(y_true_np, y_probs_np)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            y_pred_labels = (y_probs_np >= optimal_threshold).astype(np.int32)
            eval_auc = auc(fpr, tpr)
            eval_fpr, eval_tpr = fpr, tpr
        else:
            y_pred_labels = np.argmax(y_probs_np, axis=1)
            eval_auc = roc_auc_score(y_true_np, y_probs_np, multi_class='ovr')
            eval_fpr, eval_tpr = None, None

        # Compute accuracy
        eval_accuracy = accuracy_score(y_true_np, y_pred_labels)

        if debug:
            print(f"Unique predicted labels: {np.unique(y_pred_labels)}")
            print(f"Unique true labels: {np.unique(y_true_np)}")
            print(f"Confusion Matrix:\n{confusion_matrix(y_true_np, y_pred_labels)}")

        progress_bar.set_postfix_str(f"Loss = {eval_loss:.4f}, AUC = {eval_auc:.3f}")
    return eval_loss, eval_accuracy, eval_auc, y_true_np, y_pred_labels, eval_fpr, eval_tpr

def train_and_evaluate(model_fn: Callable[[], nn.Module], train_dataloader, val_dataloader, test_dataloader, num_classes: int,
                       num_epochs: int, lrs_peak_value: float = 1e-3,
                       lrs_warmup_steps: int = 5000, lrs_decay_steps: int = 50000,
                       seed: int = 42, use_ray: bool = False, debug: bool = False) -> dict:
    """
    Trains the given model on the provided dataloaders and evaluates it.

    Args:
        model_fn: A callable that returns a new instance of the model.
        train_dataloader: The dataloader for the training set.
        val_dataloader: The dataloader for the validation set.
        test_dataloader: The dataloader for the test set.
        num_classes: The number of classes.
        num_epochs: The number of epochs to train for.
        lrs_peak_value: The peak value of the learning rate schedule.
        lrs_warmup_steps: The number of warmup steps.
        lrs_decay_steps: The number of decay steps.
        seed: The seed to use for reproducibility.
        use_ray: Whether to use Ray for logging.
        debug: Whether to print extra information for debugging.

    Returns:
        metrics: A dictionary containing training and evaluation metrics.
    """
    if use_ray:
        from ray.air import session

    metrics = {
        'train_losses': [],
        'val_losses': [],
        'train_aucs': [],
        'val_aucs': [],
        'test_loss': 0.0,
        'test_accuracy': 0.0,
        'test_auc': 0.0,
        'test_classification_report': None,
        'eval_fpr': None,
        'eval_tpr': None,
    }

    print(f"\nTraining model...")
    root_key = jax.random.PRNGKey(seed=seed)
    root_key, params_key, train_key = jax.random.split(key=root_key, num=3)

    # Initialize model
    model = model_fn()

    # Get a batch to determine input shapes
    dummy_batch_spectrogram, dummy_batch_tabular, _ = next(iter(train_dataloader))
    input_shape_spectrogram = dummy_batch_spectrogram.shape[1:]
    input_shape_tabular = dummy_batch_tabular.shape[1:]
    input_dtype_spectrogram = dummy_batch_spectrogram.dtype
    input_dtype_tabular = dummy_batch_tabular.dtype
    batch_size = dummy_batch_spectrogram.shape[0]
    root_key, input_key = jax.random.split(key=root_key)
    if jnp.issubdtype(input_dtype_spectrogram, jnp.floating):
        dummy_batch_spectrogram = jax.random.uniform(key=input_key, shape=(batch_size,) + input_shape_spectrogram, dtype=input_dtype_spectrogram)
    else:
        raise ValueError(f"Unsupported dtype {input_dtype_spectrogram}")
    if jnp.issubdtype(input_dtype_tabular, jnp.floating):
        dummy_batch_tabular = jax.random.uniform(key=input_key, shape=(batch_size,) + input_shape_tabular, dtype=input_dtype_tabular)
    else:
        raise ValueError(f"Unsupported dtype {input_dtype_tabular}")

    variables = model.init(params_key, x_spectrogram=dummy_batch_spectrogram, x_tabular=dummy_batch_tabular, train=False)

    if debug:
        print(jax.tree_map(lambda x: x.shape, variables))
    print(f"Number of parameters = {sum(x.size for x in jax.tree_util.tree_leaves(variables))}")

    learning_rate_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lrs_peak_value,
        warmup_steps=lrs_warmup_steps,
        decay_steps=lrs_decay_steps,
        end_value=0.0
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=learning_rate_schedule),
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        key=train_key,
        tx=optimizer
    )

    best_val_auc, best_epoch, best_state = 0.0, 0, None
    total_train_time = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", bar_format=TQDM_BAR_FORMAT) as progress_bar:
            epoch_train_time = time.time()
            for inputs_spectrogram_batch, inputs_tabular_batch, labels_batch in train_dataloader:
                state = train_step(state, inputs_spectrogram_batch, inputs_tabular_batch, labels_batch, train_key)
                progress_bar.update(1)
            epoch_train_time = time.time() - epoch_train_time
            total_train_time += epoch_train_time

            # Evaluate on validation set
            val_loss, val_accuracy, val_auc, _, _, _, _ = evaluate(state, val_dataloader, num_classes, tqdm_desc=None, debug=debug)
            progress_bar.set_postfix_str(f"Val Loss = {val_loss:.4f}, AUC = {val_auc:.3f}, Train time = {epoch_train_time:.2f}s")

            metrics['val_losses'].append(val_loss)
            metrics['val_aucs'].append(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch + 1
                best_state = state

            if use_ray:
                session.report({'val_loss': val_loss, 'val_auc': val_auc, 'best_val_auc': best_val_auc, 'best_epoch': best_epoch})

    print(f"Best validation AUC = {best_val_auc:.3f} at epoch {best_epoch}")
    print(f"Total training time = {total_train_time:.2f}s, total time (including evaluations) = {time.time() - start_time:.2f}s")

    # Evaluate on test set using the best model
    assert best_state is not None
    test_loss, test_accuracy, test_auc, y_true, y_pred_labels, eval_fpr, eval_tpr = evaluate(best_state, test_dataloader, num_classes, tqdm_desc="Testing", debug=debug)
    metrics['test_loss'] = test_loss
    metrics['test_accuracy'] = test_accuracy
    metrics['test_auc'] = test_auc
    metrics['eval_fpr'] = eval_fpr
    metrics['eval_tpr'] = eval_tpr
    metrics['test_classification_report'] = classification_report(y_true, y_pred_labels, digits=4)

    # Print the classification report
    print("\nTest Classification Report:")
    print(metrics['test_classification_report'])

    if use_ray:
        session.report({'test_loss': test_loss, 'test_accuracy': test_accuracy, 'test_auc': test_auc})
    return metrics
