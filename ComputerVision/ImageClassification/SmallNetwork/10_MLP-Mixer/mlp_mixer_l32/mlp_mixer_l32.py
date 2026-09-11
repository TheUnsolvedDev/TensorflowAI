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