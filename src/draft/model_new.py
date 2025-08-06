import numpy as np

# Data (Jenis Gabah (ID), Kadar Air Gabah, Suhu Gabah, Suhu Ruangan, Suhu Pembakaran, Status Pengaduk)
X = np.array([[1, 28, 26, 30, 280, 1],
              [2, 27, 25, 31, 290, 1],
              [3, 26, 24, 33, 300, 0]])
y = np.array([1200, 1182, 1160])

# Langkah 1: Normalisasi (Min-Max Normalization)
def normalize(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    norm_val = (data - min_val) / (max_val - min_val)
    return norm_val, min_val, max_val

# Normalisasi fitur (X) dan target (y)
X_norm, X_min, X_max = normalize(X)
y_norm, y_min, y_max = normalize(y.reshape(-1, 1))

# Langkah 2: Cost Function (Mean Squared Error)
def compute_cost(X, y, w, b):
    m = len(X)
    predictions = np.dot(X, w) + b
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Langkah 3: Gradient Descent
def gradient_descent(X, y, w, b, learning_rate, iterations):
    m = len(X)
    for i in range(iterations):
        predictions = np.dot(X, w) + b
        errors = predictions - y
        # Turunan untuk w (bobot)
        dw = (1 / m) * np.dot(X.T, errors)
        # Turunan untuk b (bias)
        db = (1 / m) * np.sum(errors)
        # Update w dan b
        w = w - learning_rate * dw
        b = b - learning_rate * db
        # Cetak cost setiap 10000 iterasi
        if i % 10000 == 0:
            print(f"Iterasi {i}, Cost: {compute_cost(X, y, w, b)}")
            # print(f"Iterasi {i}, W: {w}, B: {b}, Cost: {compute_cost(X, y, w, b)}")
    return w, b

# Langkah 4: Prediksi dan Denormalisasi
def predict(X_new, w, b, X_min, X_max, y_min, y_max):
    # Normalisasi data baru
    X_new_norm = (X_new - X_min) / (X_max - X_min)
    # Prediksi dalam skala ternormalisasi
    y_pred_norm = np.dot(X_new_norm, w) + b
    print("Hasil Prediksi Normalisasi: ", y_pred_norm)
    # Denormalisasi ke skala asli (menit)
    y_pred = y_pred_norm * (y_max - y_min) + y_min
    return y_pred

# Inisialisasi parameter
w = np.zeros(X.shape[1])  # Bobot awal = 0
b = 0.0                   # Bias awal = 0
learning_rate = 0.01      # Learning rate
iterations = 100000        # Jumlah iterasi 100k

# Jalankan gradient descent
w, b = gradient_descent(X_norm, y_norm.ravel(), w, b, learning_rate, iterations)

# Prediksi untuk data pertama [1, 28, 26, 30, 280, 1]
X_new = np.array([1, 28, 26, 30, 280, 1])
y_pred = predict(X_new, w, b, X_min, X_max, y_min, y_max)
print(f"Estimasi durasi untuk data {X_new}: {y_pred[0]:.2f} menit")