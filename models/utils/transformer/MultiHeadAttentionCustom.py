import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class MultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, key_dim, dropout=0.1, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        assert key_dim % num_heads == 0, "key_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.key_dim = key_dim
        self.depth = key_dim // num_heads

        # 定义线性层用于生成Q, K, V
        self.Wq = layers.Dense(key_dim)
        self.Wk = layers.Dense(key_dim)
        self.Wv = layers.Dense(key_dim)

        # 最终的线性层
        self.dense = layers.Dense(key_dim)

        self.dropout = layers.Dropout(dropout)

    def split_heads(self, x, batch_size):
        """
        将最后一个维度分割成 (num_heads, depth).
        转置为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        # 生成Q, K, V
        q = self.Wq(q)  # (batch_size, seq_len, key_dim)
        k = self.Wk(k)  # (batch_size, seq_len, key_dim)
        v = self.Wv(v)  # (batch_size, seq_len, key_dim)

        # 分割成多个头
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # 计算缩放点积注意力
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch_size, num_heads, seq_len_q, seq_len_k)

        dk = tf.cast(self.depth, tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  # 添加mask

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        attention_weights = self.dropout(attention_weights)

        output = tf.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, depth)

        # 拼接所有头
        output = tf.transpose(output, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(output, (batch_size, -1, self.key_dim))  # (batch_size, seq_len_q, key_dim)

        # 最终的线性层
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, key_dim)

        return output

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout': self.dropout.rate,
        })
        return config


class TransformerEncoder(layers.Layer):
    def __init__(self, num_heads, key_dim, ff_dim, dropout=0.1, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)
        self.dropout1 = layers.Dropout(dropout)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(key_dim)
        ])
        self.dropout2 = layers.Dropout(dropout)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training, mask=None):
        # 多头自注意力
        attn_output = self.mha(x, x, x, mask)  # (batch_size, seq_len, key_dim)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, seq_len, key_dim)

        # 前馈网络
        ffn_output = self.ffn(out1)  # (batch_size, seq_len, key_dim)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, seq_len, key_dim)

        return out2

    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update({
            'num_heads': self.mha.num_heads,
            'key_dim': self.mha.key_dim,
            'ff_dim': self.ffn.layers[0].units,
            'dropout': self.dropout1.rate,
        })
        return config
