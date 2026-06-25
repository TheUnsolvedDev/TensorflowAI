import tensorflow as tf

# =========================
# Utility Layers
# =========================
class LocalResponseNormalization(tf.keras.layers.Layer):
    def __init__(self, alpha=0.0001, beta=0.75, depth_radius=5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.depth_radius = depth_radius

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(1, 1, self.channels, 1),
            initializer="ones",
            trainable=False
        )

    def call(self, x):
        squared = tf.square(x)
        window_sum = tf.nn.depthwise_conv2d(
            squared, self.kernel, strides=[1,1,1,1], padding="SAME"
        )
        norm = tf.pow(1 + self.alpha * window_sum, -self.beta)
        return x * norm


def lenet5_model(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Rescaling(1./255)(inputs)
    x = tf.keras.layers.Conv2D(6, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(16, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(120, activation='relu')(x)
    x = tf.keras.layers.Dense(84, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs, name="LeNet5")


def alexnet_model(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Rescaling(1./255)(inputs)

    x = tf.keras.layers.Conv2D(96, 11, strides=4)(x)
    x = LocalResponseNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(3, strides=2)(x)

    x = tf.keras.layers.Conv2D(256, 5, padding='same')(x)
    x = LocalResponseNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(3, strides=2)(x)

    x = tf.keras.layers.Conv2D(384, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(384, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs, name="AlexNet")


import tensorflow as tf


class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics,
                student_loss_fn,
                distillation_loss_fn,
                alpha=0.1,
                temperature=3,
                global_batch_size=32):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature
        self.global_batch_size = global_batch_size

    def call(self, inputs, training=False):
        return self.student(inputs, training=training)

    @tf.function
    def train_step(self, data):
        x, y = data

        teacher_logits = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_logits = self.student(x, training=True)

            per_example_student_loss = self.student_loss_fn(y, student_logits)

            student_loss = tf.nn.compute_average_loss(
                per_example_student_loss,
                global_batch_size=self.global_batch_size
            )

            teacher_soft = tf.nn.softmax(teacher_logits / self.temperature)
            student_soft = tf.nn.softmax(student_logits / self.temperature)

            per_example_distill_loss = self.distillation_loss_fn(
                teacher_soft, student_soft
            )

            distill_loss = tf.nn.compute_average_loss(
                per_example_distill_loss,
                global_batch_size=self.global_batch_size
            ) * (self.temperature ** 2)

            loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss

        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

        for metric in self.metrics:
            metric.update_state(y, student_logits)

        return {
            m.name: m.result() for m in self.metrics
        } | {
            "student_loss": tf.reduce_mean(per_example_student_loss),
            "distillation_loss": tf.reduce_mean(per_example_distill_loss),
            "loss": loss
        }

    @tf.function
    def test_step(self, data):
        x, y = data

        logits = self.student(x, training=False)

        per_example_loss = self.student_loss_fn(y, logits)

        loss = tf.nn.compute_average_loss(
            per_example_loss,
            global_batch_size=self.global_batch_size
        )

        for metric in self.metrics:
            metric.update_state(y, logits)

        return {
            m.name: m.result() for m in self.metrics
        } | {
            "student_loss": tf.reduce_mean(per_example_loss),
            "loss": loss
        }