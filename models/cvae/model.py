import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

class CVAE(tf.keras.Model):
    def __init__(self, signal_len=5000, channels=12, num_classes=5, latent_dim=128):
        super(CVAE, self).__init__()
        self.signal_len = signal_len
        self.channels = channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        # Input: Signal + Label
        signal_input = layers.Input(shape=(self.signal_len, self.channels))
        label_input = layers.Input(shape=(self.num_classes,))
        
        # Repeat label to match signal length and concatenate
        # (Batch, Classes) -> (Batch, 1, Classes) -> (Batch, Length, Classes)
        label_repeated = layers.RepeatVector(self.signal_len)(label_input)
        x = layers.Concatenate()([signal_input, label_repeated])
        
        # Conv Blocks
        x = layers.Conv1D(32, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv1D(64, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv1D(128, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.Flatten()(x)
        
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        
        return models.Model([signal_input, label_input], [z_mean, z_log_var], name="encoder")

    def build_decoder(self):
        # Input: Latent Z + Label
        z_input = layers.Input(shape=(self.latent_dim,))
        label_input = layers.Input(shape=(self.num_classes,))
        
        x = layers.Concatenate()([z_input, label_input])
        
        # Calculate initial shape for reshaping (signal_len / 8 because 3 strides of 2)
        # 5000 / 8 = 625
        initial_len = self.signal_len // 8
        x = layers.Dense(initial_len * 128, activation='relu')(x)
        x = layers.Reshape((initial_len, 128))(x)
        
        x = layers.Conv1DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv1DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv1DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
        
        # Output layer
        output = layers.Conv1DTranspose(self.channels, 3, padding='same', activation='linear')(x)
        
        return models.Model([z_input, label_input], output, name="decoder")

    def sample(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs):
        signal_in, label_in = inputs
        z_mean, z_log_var = self.encoder([signal_in, label_in])
        z = self.sample(z_mean, z_log_var)
        reconstruction = self.decoder([z, label_in])
        return reconstruction, z_mean, z_log_var

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0] # Unpack if (x, y)
            
        signal_in, label_in = data
        
        with tf.GradientTape() as tape:
            reconstruction, z_mean, z_log_var = self([signal_in, label_in])
            
            # Reconstruction Loss (MSE)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.mse(signal_in, reconstruction), axis=1
                )
            )
            
            # KL Divergence
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            total_loss = reconstruction_loss + kl_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss
        }
