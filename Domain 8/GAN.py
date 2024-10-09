import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate synthetic data
def generate_data(n_samples=1000, n_features=2):
    return tf.cast(np.random.normal(loc=0, scale=1, size=(n_samples, n_features)), dtype=tf.float32)

# Define the generator model
def make_generator(latent_dim, n_features):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(latent_dim,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(n_features, activation='tanh')
    ])
    return model

# Define the discriminator model
def make_discriminator(n_features):
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(n_features,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Define the GAN model
class GAN(keras.Model):
    def __init__(self, generator, discriminator, latent_dim):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        
    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(GAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn
        
    def train_step(self, real_samples):
        batch_size = tf.shape(real_samples)[0]
        
        # Train the discriminator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_samples = self.generator(random_latent_vectors)
        combined_samples = tf.concat([generated_samples, real_samples], axis=0)
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_samples)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        
        # Train the generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.zeros((batch_size, 1))
        
        with tf.GradientTape() as tape:
            generated_samples = self.generator(random_latent_vectors)
            predictions = self.discriminator(generated_samples)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        return {"d_loss": d_loss, "g_loss": g_loss}

# Prepare the dataset
X = generate_data()
dataset = tf.data.Dataset.from_tensor_slices(X).shuffle(1000).batch(32)

# Create and compile the GAN
latent_dim = 100
n_features = 2

generator = make_generator(latent_dim, n_features)
discriminator = make_discriminator(n_features)

gan = GAN(generator, discriminator, latent_dim)
gan.compile(
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss_fn=keras.losses.BinaryCrossentropy()
)

# Train the GAN
history = gan.fit(dataset, epochs=5000)

# Generate samples after training
n_samples = 1000
random_latent_vectors = tf.random.normal(shape=(n_samples, latent_dim))
generated_samples = generator.predict(random_latent_vectors)

# Visualize the results
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5, label='Real data')
plt.scatter(generated_samples[:, 0], generated_samples[:, 1], c='red', alpha=0.5, label='Generated data')
plt.legend()
plt.title('Real vs Generated Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Plot loss curves
plt.figure(figsize=(10, 6))
plt.plot(history.history['d_loss'], label='Discriminator loss')
plt.plot(history.history['g_loss'], label='Generator loss')
plt.legend()
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
