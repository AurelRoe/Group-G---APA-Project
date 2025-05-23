# custom_funtions.py
from keras.saving import register_keras_serializable
from tensorflow.keras.layers import Layer
import tensorflow as tf

@register_keras_serializable()
def focal_loss_with_penalty(y_true, y_pred, gamma=2.0, alpha=0.25, epsilon=1e-7):
    # Focal loss for multi-label classification
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)

    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    focal_loss = alpha * tf.pow(1 - p_t, gamma) * cross_entropy

    # Add penalty for all-zero predictions
    sum_pred = tf.reduce_sum(y_pred, axis=1)
    all_zero_penalty = 5.0 * tf.exp(-sum_pred)  # Penalty increases as sum approaches zero

    return tf.reduce_mean(focal_loss) + tf.reduce_mean(all_zero_penalty)

@register_keras_serializable()
def f1_metric(y_true, y_pred):
    y_pred_binary = tf.cast(tf.greater(y_pred, 0.5), tf.float32)

    true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred_binary, 1)), tf.float32))
    false_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred_binary, 1)), tf.float32))
    false_negatives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred_binary, 0)), tf.float32))

    precision = true_positives / (true_positives + false_positives + tf.keras.backend.epsilon())
    recall = true_positives / (true_positives + false_negatives + tf.keras.backend.epsilon())

    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    return f1

@register_keras_serializable()
class LearnablePositionalEncoding(Layer):
    def __init__(self, maxlen, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.pos_embedding = self.add_weight(
            shape=(maxlen, embedding_dim),
            initializer='random_normal',
            trainable=True,
            name="learnable_pos_embedding"
        )

    def call(self, x):
        # x shape: (batch_size, sequence_length, embedding_dim)
        seq_len = tf.shape(x)[1]
        return x + self.pos_embedding[tf.newaxis, :seq_len, :]