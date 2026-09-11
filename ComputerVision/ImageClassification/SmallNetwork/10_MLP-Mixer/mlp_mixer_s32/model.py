import tensorflow as tf


class MLP(tf.keras.layers.Layer):

    def __init__(self, hidden_dim, output_dim, dropout=0.0):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_dim, activation=tf.nn.gelu)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.fc2 = tf.keras.layers.Dense(output_dim)
        self.drop2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.drop1(x, training=training)
        x = self.fc2(x)
        x = self.drop2(x, training=training)
        return x


class MixerBlock(tf.keras.layers.Layer):

    def __init__(self, num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim, dropout=0.0):
        super().__init__()

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.token_mlp = MLP(tokens_mlp_dim, num_patches, dropout)

        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.channel_mlp = MLP(channels_mlp_dim, hidden_dim, dropout)

    def call(self, x, training=False):

        y = self.norm1(x)

        y = tf.transpose(y, [0, 2, 1])
        y = self.token_mlp(y, training)
        y = tf.transpose(y, [0, 2, 1])

        x = x + y

        y = self.norm2(x)
        y = self.channel_mlp(y, training)

        return x + y