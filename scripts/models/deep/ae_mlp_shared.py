"""
Shared TensorFlow/Keras utilities for AE-MLP model scripts.

Contains the AE-MLP model builder, TF dataset creation, GPU setup,
and random seed utilities used across multiple AE-MLP training scripts.

Placed in deep/ (not common/) to avoid adding a TensorFlow dependency
to the framework-agnostic common module.
"""

import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras import mixed_precision


def set_random_seed(seed: int):
    """Set random seeds for reproducibility (TensorFlow variant)."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"    Random seed set to: {seed}")


def create_tf_dataset(X, y, batch_size, shuffle=True, prefetch=True):
    """Create optimized tf.data.Dataset with data pinned to CPU."""
    with tf.device('/CPU:0'):
        outputs = {
            'decoder': X.astype(np.float32),
            'ae_action': y.astype(np.float32),
            'action': y.astype(np.float32),
        }

        dataset = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), outputs))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(len(X), 50000))

        dataset = dataset.batch(batch_size)

        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def build_ae_mlp_model(params: dict) -> Model:
    """Build the AE-MLP (AutoEncoder + MLP) model.

    Architecture: Input -> BN -> GaussianNoise -> Encoder -> Decoder + AE-branch + Main MLP

    Args:
        params: dict with keys:
            - num_columns: input feature count
            - hidden_units: list of hidden layer sizes
            - dropout_rates: list of dropout rates
            - lr: learning rate
            - loss_weights: dict of loss weights for decoder/ae_action/action
    """
    num_columns = params['num_columns']
    hidden_units = params['hidden_units']
    dropout_rates = params['dropout_rates']
    lr = params['lr']
    loss_weights = params['loss_weights']

    inp = layers.Input(shape=(num_columns,), name='input')

    # Input normalization
    x0 = layers.BatchNormalization(name='input_bn')(inp)

    # Encoder
    encoder = layers.GaussianNoise(dropout_rates[0], name='noise')(x0)
    encoder = layers.Dense(hidden_units[0], name='encoder_dense')(encoder)
    encoder = layers.BatchNormalization(name='encoder_bn')(encoder)
    encoder = layers.Activation('swish', name='encoder_act')(encoder)

    # Decoder (reconstruct original input)
    decoder = layers.Dropout(dropout_rates[1], name='decoder_dropout')(encoder)
    decoder = layers.Dense(num_columns, dtype='float32', name='decoder')(decoder)

    # Auxiliary prediction branch (based on decoder output)
    x_ae = layers.Dense(hidden_units[1], name='ae_dense1')(decoder)
    x_ae = layers.BatchNormalization(name='ae_bn1')(x_ae)
    x_ae = layers.Activation('swish', name='ae_act1')(x_ae)
    x_ae = layers.Dropout(dropout_rates[2], name='ae_dropout1')(x_ae)
    out_ae = layers.Dense(1, dtype='float32', name='ae_action')(x_ae)

    # Main branch: original features + encoder features
    x = layers.Concatenate(name='concat')([x0, encoder])
    x = layers.BatchNormalization(name='main_bn0')(x)
    x = layers.Dropout(dropout_rates[3], name='main_dropout0')(x)

    # MLP body
    for i in range(2, len(hidden_units)):
        dropout_idx = min(i + 2, len(dropout_rates) - 1)
        x = layers.Dense(hidden_units[i], name=f'main_dense{i-1}')(x)
        x = layers.BatchNormalization(name=f'main_bn{i-1}')(x)
        x = layers.Activation('swish', name=f'main_act{i-1}')(x)
        x = layers.Dropout(dropout_rates[dropout_idx], name=f'main_dropout{i-1}')(x)

    # Main output (float32 for numerical stability)
    out = layers.Dense(1, dtype='float32', name='action')(x)

    model = Model(inputs=inp, outputs=[decoder, out_ae, out], name='AE_MLP')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss={
            'decoder': 'mse',
            'ae_action': 'mse',
            'action': 'mse',
        },
        loss_weights=loss_weights,
    )

    return model


def setup_gpu(gpu: int, use_mixed_precision: bool = False):
    """Configure GPU visibility and optional mixed precision."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpu >= 0 and gpus:
        try:
            tf.config.set_visible_devices(gpus[gpu], 'GPU')
            print(f"    Using GPU: {gpus[gpu]}")

            if use_mixed_precision:
                mixed_precision.set_global_policy('mixed_float16')
                print("    Mixed precision (FP16) enabled")
        except RuntimeError as e:
            print(f"    GPU setup error: {e}")
    else:
        tf.config.set_visible_devices([], 'GPU')
        print("    Using CPU")
