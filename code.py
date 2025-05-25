#code
new model # model.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ks_2samp, skew, kurtosis
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load dataset locally
csv_file = 'gandataEUROSTOXX.csv'  # Adjust path if needed, e.g., 'TBC-GAN\gandataEUROSTOXX.csv'
csv_path = os.path.join('C:', 'Users', 'hp', csv_file)  # Full path: C:\Users\hp\gandataEUROSTOXX.csv
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Cannot find {csv_path}. Ensure gandataEUROSTOXX.csv is in C:\\Users\\hp")
print(f"Loading {csv_path}")
data = pd.read_csv(csv_path)

# Preprocess data
if 'Indexvalue' not in data.columns:
    print("Column 'Indexvalue' not found. Available columns:", data.columns)
    raise ValueError("Please specify the correct column name")
values = data['Indexvalue'].values.reshape(-1, 1)
if len(values) != 2206:
    print(f"Warning: Expected 2206 entries, found {len(values)}")
scaler = MinMaxScaler()
values_scaled = scaler.fit_transform(values)

# Create sequences
seq_len = 24
n_features = 1
sequences = np.array([values_scaled[i:i+seq_len] for i in range(len(values_scaled) - seq_len + 1)])
X = sequences  # Shape: (2183, 24, 1)
print(f"Generated {X.shape[0]} sequences")

# Model parameters
hidden_dim = 16
num_layers = 1
batch_size = 128
iterations = 1000
n_critic = 5
n_sequences = X.shape[0]

# Transformer Block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, num_heads=2, ff_dim=32):
        super(TransformerBlock, self).__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(hidden_dim)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dropout2 = tf.keras.layers.Dropout(0.2)

    def call(self, inputs, training=True):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Generator
def build_generator():
    inputs = tf.keras.layers.Input(shape=(seq_len, n_features))
    x = inputs
    for _ in range(num_layers):
        x = TransformerBlock(hidden_dim)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_dim, return_sequences=True))(x)
    x = tf.keras.layers.Conv1D(filters=hidden_dim, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=n_features, kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Lambda(lambda x: x * 0.2)(x)
    model = tf.keras.Model(inputs, x)
    return model

# Discriminator
def build_discriminator():
    inputs = tf.keras.layers.Input(shape=(seq_len, n_features))
    x = inputs
    for _ in range(num_layers):
        x = TransformerBlock(hidden_dim)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_dim, return_sequences=True))(x)
    x = tf.keras.layers.Conv1D(filters=hidden_dim, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, x)
    return model

# Embedder
def build_embedder():
    inputs = tf.keras.layers.Input(shape=(seq_len, n_features))
    x = inputs
    for _ in range(num_layers):
        x = TransformerBlock(hidden_dim)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_dim, return_sequences=True))(x)
    x = tf.keras.layers.Dense(hidden_dim)(x)
    model = tf.keras.Model(inputs, x)
    return model

# Supervisor
def build_supervisor():
    inputs = tf.keras.layers.Input(shape=(seq_len, hidden_dim))
    x = inputs
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_dim, return_sequences=True))(x)
    x = tf.keras.layers.Conv1D(filters=hidden_dim, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=n_features, kernel_size=3, padding='same')(x)
    model = tf.keras.Model(inputs, x)
    return model

# Build models
generator = build_generator()
discriminator = build_discriminator()
embedder = build_embedder()
supervisor = build_supervisor()

# Optimizers
g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5)
s_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5)

# WGAN-GP Loss
def gradient_penalty(real_data, fake_data, discriminator):
    batch_size = tf.shape(real_data)[0]
    real_data = tf.cast(real_data, tf.float32)
    fake_data = tf.cast(fake_data, tf.float32)
    alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0, dtype=tf.float32)
    interpolated = alpha * real_data + (1.0 - alpha) * fake_data
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)
    grads = tape.gradient(pred, interpolated)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]) + 1e-10)
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

# Training step
@tf.function
def train_step(real_data):
    real_data = tf.cast(real_data, tf.float32)
    batch_size = tf.shape(real_data)[0]
    noise = tf.random.normal([batch_size, seq_len, n_features], dtype=tf.float32)
    
    d_loss = 0.0
    for _ in range(n_critic):
        with tf.GradientTape() as d_tape:
            fake_data = generator(noise, training=True)
            fake_data = tf.cast(fake_data, tf.float32)
            real_output = discriminator(real_data, training=True)
            fake_output = discriminator(fake_data, training=True)
            d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
            gp = gradient_penalty(real_data, fake_data, discriminator)
            d_loss += 2.0 * gp
        d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
        for var in discriminator.trainable_variables:
            var.assign(tf.clip_by_value(var, -0.01, 0.01))
    
    with tf.GradientTape() as g_tape, tf.GradientTape() as s_tape:
        fake_data = generator(noise, training=True)
        fake_data = tf.cast(fake_data, tf.float32)
        fake_output = discriminator(fake_data, training=True)
        g_loss_gan = -tf.reduce_mean(fake_output)
        embedded_real = embedder(real_data, training=True)
        supervised_output = supervisor(embedded_real, training=True)
        s_loss = tf.keras.losses.MeanSquaredError()(real_data, supervised_output)
        g_loss = g_loss_gan + 10.0 * s_loss
    
    g_grads = g_tape.gradient(g_loss, generator.trainable_variables + supervisor.trainable_variables)
    s_grads = s_tape.gradient(s_loss, embedder.trainable_variables + supervisor.trainable_variables)
    
    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables + supervisor.trainable_variables))
    s_optimizer.apply_gradients(zip(s_grads, embedder.trainable_variables + supervisor.trainable_variables))
    
    return d_loss, g_loss, fake_data

# Training loop
dataset = tf.data.Dataset.from_tensor_slices(X).shuffle(1000).batch(batch_size)
for iteration in range(iterations):
    d_loss_total = 0.0
    g_loss_total = 0.0
    batch_count = 0
    for batch in dataset:
        d_loss, g_loss, fake_data = train_step(batch)
        d_loss_total += d_loss
        g_loss_total += g_loss
        batch_count += 1
    avg_d_loss = d_loss_total / batch_count
    avg_g_loss = g_loss_total / batch_count
    if (iteration + 1) % 200 == 0:
        fake_variance = np.var(fake_data.numpy())
        print(f"Iteration {iteration + 1}, D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}, Fake Variance: {fake_variance:.6f}")
        if fake_variance < 1e-6:
            print("Warning: Low fake data variance detected")

# Generate synthetic data
noise = tf.random.normal([n_sequences, seq_len, n_features], dtype=tf.float32)
synthetic_data = generator(noise, training=False).numpy()
print(f"Synthetic data variance: {np.var(synthetic_data):.6f}")

# AR(2) baseline
model = AutoReg(values_scaled[:,0], lags=2)
model_fit = model.fit()
synthetic_ar_sequences = []
for _ in range(n_sequences):
    start_idx = np.random.randint(0, len(values_scaled) - seq_len)
    initial = values_scaled[start_idx:start_idx+2, 0]
    synthetic_seq = list(initial)
    for t in range(2, seq_len):
        pred = model_fit.params[0] + model_fit.params[1] * synthetic_seq[-1] + model_fit.params[2] * synthetic_seq[-2]
        noise = np.random.normal(0, np.std(values_scaled[:,0]) * 0.1)
        next_val = np.clip(pred + noise, 0, 1)
        synthetic_seq.append(next_val)
    synthetic_ar_sequences.append(synthetic_seq)
synthetic_ar_sequences = np.array(synthetic_ar_sequences).reshape(-1, seq_len, 1)

# Compute evaluation metrics
def compute_metrics(real_seqs, synthetic_seqs, lag=10):
    real_flat = real_seqs.reshape(-1)
    synthetic_flat = np.clip(synthetic_seqs.reshape(-1), -1, 1)
    ks_stat, _ = ks_2samp(real_flat, synthetic_flat)
    
    if np.var(synthetic_flat) < 1e-6:
        print("Warning: Synthetic data has near-zero variance")
        return ks_stat, np.nan, np.nan, np.nan
    
    skew_diff = abs(skew(real_flat, nan_policy='omit') - skew(synthetic_flat, nan_policy='omit'))
    kurt_diff = abs(kurtosis(real_flat, nan_policy='omit') - kurtosis(synthetic_flat, nan_policy='omit'))
    
    try:
        acf_real = np.mean([acf(seq[:,0], nlags=lag, fft=False) for seq in real_seqs], axis=0)
        acf_synthetic = np.mean([acf(seq[:,0], nlags=lag, fft=False) for seq in synthetic_seqs], axis=0)
        acf_mse = np.mean((acf_real - acf_synthetic)**2)
    except Exception as e:
        print(f"Warning: ACF computation failed: {e}")
        acf_mse = np.nan
    
    return ks_stat, skew_diff, kurt_diff, acf_mse

# Compute metrics
real_sequences = X
tbcgan_metrics = compute_metrics(real_sequences, synthetic_data)
ar_metrics = compute_metrics(real_sequences, synthetic_ar_sequences)

# Create table
print("\nTable 1: Model Comparison (TBC-GAN on EURO STOXX 50 Volatility)")
print("Model\tKS Statistic\tSkewness Diff\tKurtosis Diff\tACF MSE")
print(f"TBC-GAN\t{tbcgan_metrics[0]:.4f}\t\t{tbcgan_metrics[1]:.4f}\t\t{tbcgan_metrics[2]:.4f}\t\t{tbcgan_metrics[3]:.4f}")
print(f"AR(2)\t{ar_metrics[0]:.4f}\t\t{ar_metrics[1]:.4f}\t\t{ar_metrics[2]:.4f}\t\t{ar_metrics[3]:.4f}")

# Visualizations
plt.figure(figsize=(10, 6))
sns.kdeplot(real_sequences.reshape(-1), label='Real', color='blue', warn_singular=False)
sns.kdeplot(synthetic_data.reshape(-1), label='TBC-GAN', color='red', warn_singular=False)
sns.kdeplot(synthetic_ar_sequences.reshape(-1), label='AR(2)', color='green', warn_singular=False)
plt.legend()
plt.title('Empirical Density Comparison (EURO STOXX 50 Volatility)')
plt.xlabel('Normalized Volatility Index')
plt.ylabel('Density')
plt.show()

lag = 10
try:
    acf_real = np.mean([acf(seq[:,0], nlags=lag, fft=False) for seq in real_sequences], axis=0)
    acf_tbcgan = np.mean([acf(seq[:,0], nlags=lag, fft=False) for seq in synthetic_data], axis=0)
    acf_ar = np.mean([acf(seq[:,0], nlags=lag, fft=False) for seq in synthetic_ar_sequences], axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(acf_real, label='Real', color='blue')
    plt.plot(acf_tbcgan, label='TBC-GAN', color='red')
    plt.plot(acf_ar, label='AR(2)', color='green')
    plt.legend()
    plt.title('Average Autocorrelation Function (ACF)')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.show()
except Exception as e:
    print(f"Warning: ACF plot failed: {e}")

plt.figure(figsize=(10, 6))
plt.plot(real_sequences[0,:,0], label='Real', color='blue')
plt.plot(synthetic_data[0,:,0], label='TBC-GAN', color='red')
plt.plot(synthetic_ar_sequences[0,:,0], label='AR(2)', color='green')
plt.legend()
plt.title('Sample Path Comparison (EURO STOXX 50 Volatility)')
plt.xlabel('Time Step')
plt.ylabel('Normalized Volatility Index')
plt.show()
