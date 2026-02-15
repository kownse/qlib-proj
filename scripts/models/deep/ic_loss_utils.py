"""
IC Loss utilities for TensorFlow/Keras models.

Provides:
1. tf_ic_loss - Batch-level Pearson IC loss (Keras-compatible)
2. make_mixed_ic_mse_loss - Factory for MSE + IC_weight * (-IC) loss
3. ICEarlyStoppingCallback - Early stopping based on cross-sectional IC

Reference implementations:
  - kan_model.py:545 (_ic_loss) - PyTorch batch-level IC
  - transformer_model.py:142 (ic_loss) - PyTorch batch-level IC
  - cv_utils.py:198 (compute_ic) - Date-grouped IC calculation
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


def tf_ic_loss(y_true, y_pred):
    """Batch-level Pearson IC loss (negative correlation) for Keras.

    Computes -IC where IC = Pearson correlation between predictions and labels.
    Minimizing this loss maximizes the information coefficient.

    Batch-level IC is an approximation (mixes dates), but has been validated
    in KAN and Transformer models as effective for training.

    Args:
        y_true: Ground truth values, shape (batch_size, 1) or (batch_size,)
        y_pred: Predicted values, shape (batch_size, 1) or (batch_size,)

    Returns:
        Scalar tensor: -IC (negative Pearson correlation)
    """
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

    pred_centered = y_pred - tf.reduce_mean(y_pred)
    label_centered = y_true - tf.reduce_mean(y_true)

    pred_std = tf.math.reduce_std(y_pred)
    label_std = tf.math.reduce_std(y_true)

    # Return 0 when variance is too low (degenerate batch)
    safe = tf.logical_and(pred_std > 1e-8, label_std > 1e-8)
    cov = tf.reduce_mean(pred_centered * label_centered)
    ic = cov / (pred_std * label_std + 1e-12)

    return tf.where(safe, -ic, 0.0)


def make_mixed_ic_mse_loss(ic_weight):
    """Factory function for mixed MSE + IC loss.

    Args:
        ic_weight: Weight for IC loss component.
            0.0 -> returns plain 'mse' string (backward compatible)
            >0  -> returns custom loss: MSE + ic_weight * (-IC)

    Returns:
        'mse' string or a Keras-compatible loss function
    """
    if ic_weight == 0.0 or ic_weight is None:
        return 'mse'

    def mixed_ic_mse_loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        ic_loss = tf_ic_loss(y_true, y_pred)
        return mse + ic_weight * ic_loss

    mixed_ic_mse_loss.__name__ = f'mixed_ic_mse_w{ic_weight}'
    return mixed_ic_mse_loss


class ICEarlyStoppingCallback(keras.callbacks.Callback):
    """Early stopping based on cross-sectional IC on the validation set.

    Instead of monitoring MSE-based val_loss (which favors mean predictions),
    this callback computes the actual evaluation metric (IC) after each epoch
    and uses it for early stopping with best weight restoration.

    The IC is computed by grouping predictions by date and calculating
    per-day Pearson correlations, following the same logic as cv_utils.compute_ic.

    Args:
        X_valid: Validation features array
        y_valid: Validation labels for the primary horizon (1D array)
        valid_index: MultiIndex with 'datetime' level for date grouping
        primary_output_idx: Index of primary horizon output in model.predict() list
        patience: Number of epochs with no improvement before stopping
        batch_size: Batch size for prediction
        min_delta: Minimum improvement to qualify as better
        verbose: Verbosity level (0=silent, 1=progress)
    """

    def __init__(self, X_valid, y_valid, valid_index,
                 primary_output_idx=2, patience=10,
                 batch_size=4096, min_delta=0.0, verbose=1):
        super().__init__()
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.valid_index = valid_index
        self.primary_output_idx = primary_output_idx
        self.patience = patience
        self.batch_size = batch_size
        self.min_delta = min_delta
        self.verbose = verbose

        self.best_ic = -np.inf
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        self.best_epoch = 0

        # Pre-compute date grouping for efficient IC calculation
        if hasattr(valid_index, 'get_level_values'):
            self._dates = valid_index.get_level_values('datetime')
        else:
            self._dates = None

    def _compute_validation_ic(self):
        """Compute cross-sectional IC on the full validation set."""
        preds = self.model.predict(self.X_valid, batch_size=self.batch_size, verbose=0)
        pred = preds[self.primary_output_idx].flatten()

        if self._dates is None:
            # Fallback: batch-level correlation
            corr = np.corrcoef(pred, self.y_valid)[0, 1]
            return corr if not np.isnan(corr) else 0.0, 0.0

        df = pd.DataFrame({
            'pred': pred,
            'label': self.y_valid,
        }, index=self.valid_index)

        ic_by_date = df.groupby(level='datetime').apply(
            lambda x: x['pred'].corr(x['label']) if len(x) > 1 else np.nan
        )
        ic_by_date = ic_by_date.dropna()

        if len(ic_by_date) == 0:
            return 0.0, 0.0

        mean_ic = ic_by_date.mean()
        ic_std = ic_by_date.std()
        return mean_ic, ic_std

    def on_train_begin(self, logs=None):
        self.best_ic = -np.inf
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        mean_ic, ic_std = self._compute_validation_ic()
        icir = mean_ic / ic_std if ic_std > 0 else 0.0

        # Inject into logs for history recording
        if logs is not None:
            logs['val_ic'] = mean_ic
            logs['val_icir'] = icir

        if mean_ic > self.best_ic + self.min_delta:
            self.best_ic = mean_ic
            self.best_weights = self.model.get_weights()
            self.wait = 0
            self.best_epoch = epoch + 1
            if self.verbose > 0:
                print(f'  IC EarlyStopping: epoch {epoch+1}, val_ic={mean_ic:.4f}, '
                      f'icir={icir:.4f} (new best)')
        else:
            self.wait += 1
            if self.verbose > 0:
                print(f'  IC EarlyStopping: epoch {epoch+1}, val_ic={mean_ic:.4f}, '
                      f'icir={icir:.4f} (patience {self.wait}/{self.patience})')
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.best_weights is not None:
                    if self.verbose > 0:
                        print(f'  Restoring best weights from epoch {self.best_epoch} '
                              f'(IC={self.best_ic:.4f})')
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f'  IC EarlyStopping: stopped at epoch {self.stopped_epoch + 1}, '
                  f'best epoch {self.best_epoch} with IC={self.best_ic:.4f}')
