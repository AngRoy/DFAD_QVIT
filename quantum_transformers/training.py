import time
from typing import Optional, Callable

import numpy as np
import numpy.typing as npt
import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.training.train_state
import optax
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report
from tqdm import tqdm

TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'

class TrainState(flax.training.train_state.TrainState):
    # See https://flax.readthedocs.io/en/latest/guides/dropout.html.
    key: jax.random.KeyArray  # type: ignore

@jax.jit
def train_step(state: TrainState, inputs: jax.Array, labels: jax.Array, key: jax.random.KeyArray) -> TrainState:
    """
    Performs a single training step on the given batch of inputs and labels.

    Args:
        state: The current training state.
        inputs: The batch of inputs.
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
            x=inputs,
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
def eval_step(state: TrainState, inputs: jax.Array, labels: jax.Array) -> tuple[float, jax.Array]:
    """
    Performs a single evaluation step on the given batch of inputs and labels.

    Args:
        state: The current training state.
        inputs: The batch of inputs.
        labels: The batch of labels.

    Returns:
        loss: The loss on the given batch.
        logits: The logits on the given batch.
    """
    logits = state.apply_fn(
        {'params': state.params},
        x=inputs,
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
        for inputs_batch, labels_batch in eval_dataloader:
            loss_batch, logits_batch = eval_step(state, inputs_batch, labels_batch)
            logits_list.append(logits_batch)
            labels_list.append(labels_batch)
            eval_loss += loss_batch
            progress_bar.update(1)
        eval_loss /= len(eval_dataloader)
        logits = jnp.concatenate(logits_list)
        y_true = jnp.concatenate(labels_list)
        if debug:
            print(f"logits = {logits}")

        if num_classes == 2:
            y_probs = jax.nn.sigmoid(logits)
            y_pred_labels = (y_probs >= 0.5).astype(jnp.int32)
        else:
            y_probs = jax.nn.softmax(logits, axis=1)
            y_pred_labels = jnp.argmax(y_probs, axis=1)

        if debug:
            print(f"y_probs = {y_probs}")
            print(f"y_pred_labels = {y_pred_labels}")
            print(f"y_true = {y_true}")

        # Compute accuracy
        eval_accuracy = jnp.mean(y_pred_labels == y_true)
        # Compute AUC and ROC curve
        if num_classes == 2:
            y_true_np = np.array(y_true)
            y_probs_np = np.array(y_probs)
            eval_fpr, eval_tpr, _ = roc_curve(y_true_np, y_probs_np)
            eval_auc = auc(eval_fpr, eval_tpr)
        else:
            eval_fpr, eval_tpr = None, None
            eval_auc = roc_auc_score(y_true, y_probs, multi_class='ovr')

        progress_bar.set_postfix_str(f"Loss = {eval_loss:.4f}, Acc = {eval_accuracy:.3f}, AUC = {eval_auc:.3f}")
    return eval_loss, eval_accuracy, eval_auc, y_true, y_pred_labels, eval_fpr, eval_tpr

def evaluate_ensemble(states: list[TrainState], eval_dataloader, num_classes: int,
                      tqdm_desc: Optional[str] = "Testing Ensemble", debug: bool = False) -> tuple[float, float, float, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
    """
    Evaluates an ensemble of models on the given dataloader.

    Args:
        states: A list of TrainState instances for each model in the ensemble.
        eval_dataloader: The dataloader to evaluate on.
        num_classes: The number of classes.
        tqdm_desc: The description to use for the tqdm progress bar.
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
    ensemble_logits_list = []
    labels_list = []
    eval_loss = 0.0

    with tqdm(total=len(eval_dataloader), desc=tqdm_desc, unit="batch", bar_format=TQDM_BAR_FORMAT, disable=tqdm_desc is None) as progress_bar:
        for inputs_batch, labels_batch in eval_dataloader:
            batch_logits = []
            batch_loss = 0.0
            for state in states:
                loss_batch, logits_batch = eval_step(state, inputs_batch, labels_batch)
                batch_logits.append(logits_batch)
                batch_loss += loss_batch
            # Average loss over ensemble
            eval_loss += batch_loss / len(states)
            # Stack logits from each model and average them
            avg_logits = jnp.mean(jnp.stack(batch_logits), axis=0)
            ensemble_logits_list.append(avg_logits)
            labels_list.append(labels_batch)
            progress_bar.update(1)
        eval_loss /= len(eval_dataloader)
        logits = jnp.concatenate(ensemble_logits_list)
        y_true = jnp.concatenate(labels_list)

        if num_classes == 2:
            y_probs = jax.nn.sigmoid(logits)
            y_pred_labels = (y_probs >= 0.5).astype(jnp.int32)
        else:
            y_probs = jax.nn.softmax(logits, axis=1)
            y_pred_labels = jnp.argmax(y_probs, axis=1)

        eval_accuracy = jnp.mean(y_pred_labels == y_true)

        # Compute AUC and ROC curve
        if num_classes == 2:
            y_true_np = np.array(y_true)
            y_probs_np = np.array(y_probs)
            eval_fpr, eval_tpr, _ = roc_curve(y_true_np, y_probs_np)
            eval_auc = auc(eval_fpr, eval_tpr)
        else:
            eval_fpr, eval_tpr = None, None
            eval_auc = roc_auc_score(y_true, y_probs, multi_class='ovr')

        progress_bar.set_postfix_str(f"Loss = {eval_loss:.4f}, Acc = {eval_accuracy:.3f}, AUC = {eval_auc:.3f}")

    return eval_loss, eval_accuracy, eval_auc, y_true, y_pred_labels, eval_fpr, eval_tpr

def train_and_evaluate(model_fn: Callable[[], nn.Module], train_dataloader, val_dataloader, test_dataloader, num_classes: int,
                       num_epochs: int, ensemble_size: int = 1, lrs_peak_value: float = 1e-3,
                       lrs_warmup_steps: int = 5000, lrs_decay_steps: int = 50000,
                       seed: int = 42, use_ray: bool = False, debug: bool = False) -> dict:
    """
    Trains the given model(s) on the provided dataloaders and evaluates them.

    Args:
        model_fn: A callable that returns a new instance of the model.
        train_dataloader: The dataloader for the training set.
        val_dataloader: The dataloader for the validation set.
        test_dataloader: The dataloader for the test set.
        num_classes: The number of classes.
        num_epochs: The number of epochs to train for.
        ensemble_size: The number of models to train for ensembling.
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

    ensemble_states = []
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': [],
        'train_aucs': [],
        'val_aucs': [],
        'test_loss': 0.0,
        'test_accuracy': 0.0,
        'test_auc': 0.0,
        'test_classification_report': None,
        'eval_fpr': None,
        'eval_tpr': None,
    }

    for ensemble_idx in range(ensemble_size):
        print(f"\nTraining model {ensemble_idx + 1}/{ensemble_size} for ensembling...")
        root_key = jax.random.PRNGKey(seed=seed + ensemble_idx)
        root_key, params_key, train_key = jax.random.split(key=root_key, num=3)

        # Initialize model
        model = model_fn()

        dummy_batch = next(iter(train_dataloader))[0]
        input_shape = dummy_batch.shape[1:]
        input_dtype = dummy_batch.dtype
        batch_size = dummy_batch.shape[0]
        root_key, input_key = jax.random.split(key=root_key)
        if jnp.issubdtype(input_dtype, jnp.floating):
            dummy_batch = jax.random.uniform(key=input_key, shape=(batch_size,) + input_shape, dtype=input_dtype)
        elif jnp.issubdtype(input_dtype, jnp.integer):
            dummy_batch = jax.random.randint(key=input_key, shape=(batch_size,) + input_shape, minval=0, maxval=100, dtype=input_dtype)
        else:
            raise ValueError(f"Unsupported dtype {input_dtype}")

        variables = model.init(params_key, dummy_batch, train=False)

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
            with tqdm(total=len(train_dataloader), desc=f"Model {ensemble_idx+1}, Epoch {epoch+1}/{num_epochs}", unit="batch", bar_format=TQDM_BAR_FORMAT) as progress_bar:
                epoch_train_time = time.time()
                for inputs_batch, labels_batch in train_dataloader:
                    state = train_step(state, inputs_batch, labels_batch, train_key)
                    progress_bar.update(1)
                epoch_train_time = time.time() - epoch_train_time
                total_train_time += epoch_train_time

                train_loss, train_accuracy, train_auc, _, _, _, _ = evaluate(state, train_dataloader, num_classes, tqdm_desc=None, debug=debug)
                val_loss, val_accuracy, val_auc, _, _, _, _ = evaluate(state, val_dataloader, num_classes, tqdm_desc=None, debug=debug)
                progress_bar.set_postfix_str(f"Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy:.3f}, Val AUC = {val_auc:.3f}, Train time = {epoch_train_time:.2f}s")

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_epoch = epoch + 1
                    best_state = state

                if use_ray:
                    session.report({'val_loss': val_loss, 'val_accuracy': val_accuracy, 'val_auc': val_auc, 'best_val_auc': best_val_auc, 'best_epoch': best_epoch})

        print(f"Best validation AUC for model {ensemble_idx+1} = {best_val_auc:.3f} at epoch {best_epoch}")
        print(f"Total training time for model {ensemble_idx+1} = {total_train_time:.2f}s, total time (including evaluations) = {time.time() - start_time:.2f}s")

        ensemble_states.append(best_state)

    # Evaluate ensemble on test set
    test_loss, test_accuracy, test_auc, y_true, y_pred_labels, eval_fpr, eval_tpr = evaluate_ensemble(ensemble_states, test_dataloader, num_classes, debug=debug)

    # Generate classification report
    metrics['test_loss'] = test_loss
    metrics['test_accuracy'] = test_accuracy
    metrics['test_auc'] = test_auc
    metrics['eval_fpr'] = eval_fpr
    metrics['eval_tpr'] = eval_tpr
    metrics['test_classification_report'] = classification_report(y_true, y_pred_labels, digits=4)
    print("\nTest Classification Report:")
    print(metrics['test_classification_report'])

    if use_ray:
        session.report({'test_loss': test_loss, 'test_accuracy': test_accuracy, 'test_auc': test_auc})
    return metrics
