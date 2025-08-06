import numpy as np
import logging
import requests

# Logging untuk mencatat info saat training dan debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionModel:
    def __init__(self, api_url="http://127.0.0.1:3333/api/dataset"):
        # Inisialisasi bobot dan bias
        self.w = None
        self.b = None
        self.X_min = None
        self.X_max = None
        self.y_min = None
        self.y_max = None
        self.training_data = []

        # Ambil data dan siapkan training
        self.load_training_data(api_url)
        self.normalize_training_data()
        self.initialize_weights()
        self.train_model()
        self.evaluate_model()

    def load_training_data(self, api_url):
        try:
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            raw_data = response.json()
            logger.info(f"Loaded {len(raw_data)} dataset entries from API")

            for process in raw_data:
                grain_type_id = float(process['grain_type_id'])
                weight = float(process['berat_gabah'])
                intervals = process['intervals']

                for interval in intervals:
                    measurements = interval['sensor_data']
                    drying_time = float(interval['estimasi_durasi']) if interval['estimasi_durasi'] is not None else 0.0

                    if not measurements or len(measurements) != 1:
                        logger.warning(f"Invalid sensor data in interval {interval['interval_id']} (Process ID: {process['process_id']}, DryingTime: {drying_time})")
                        continue

                    measurement = measurements[0]
                    if any(measurement[key] is None for key in ['suhu_gabah', 'kadar_air_gabah', 'suhu_ruangan', 'suhu_pembakaran', 'status_pengaduk']):
                        logger.warning(f"Missing required sensor data in interval {interval['interval_id']} (Process ID: {process['process_id']}, DryingTime: {drying_time})")
                        continue

                    data = {
                        "GrainTypeId": grain_type_id,
                        "GrainMoisture": float(measurement['kadar_air_gabah']),
                        "GrainTemperature": float(measurement['suhu_gabah']),
                        "RoomTemperature": float(measurement['suhu_ruangan']),
                        "BurningTemperature": float(measurement['suhu_pembakaran']),
                        "StirrerStatus": float(measurement['status_pengaduk']),
                        "Weight": weight,
                        "DryingTime": drying_time
                    }
                    self.training_data.append(data)

            logger.info(f"Processed {len(self.training_data)} training intervals")
        except requests.RequestException as e:
            logger.error(f"Error loading training data from API: {e}")
            raise
        except (ValueError, KeyError) as e:
            logger.error(f"Error processing API data: {e}")
            raise

    def normalize(self, data):
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0  # Hindari pembagian dengan nol
        norm_val = (data - min_val) / range_val
        return norm_val, min_val, max_val

    def normalize_training_data(self):
        if not self.training_data:
            logger.error("No training data available")
            raise ValueError("Training data is empty")

        X = np.array([
            [d['GrainTypeId'], d['GrainMoisture'], d['GrainTemperature'], 
             d['RoomTemperature'], d['BurningTemperature'], d['StirrerStatus'], d['Weight']]
            for d in self.training_data
        ])
        y = np.array([d['DryingTime'] for d in self.training_data])

        self.X_norm, self.X_min, self.X_max = self.normalize(X)
        self.y_norm, self.y_min, self.y_max = self.normalize(y.reshape(-1, 1))
        self.y_norm = self.y_norm.ravel()  # Flatten untuk gradient descent
        self.y_actual = y  # Simpan nilai aktual untuk evaluasi

    def initialize_weights(self):
        self.w = np.zeros(7)  # 7 fitur
        self.b = 0.0
        logger.info(f"Initialized weights: {[round(w, 3) for w in self.w]}, bias: {self.b}")

    def compute_cost(self, X, y):
        m = X.shape[0]
        predictions = np.dot(X, self.w) + self.b
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost

    def gradient_descent(self, X, y, w, b, learning_rate=0.01, iterations=50000):
        m = len(X)
        for i in range(iterations):
            predictions = np.dot(X, w) + b
            errors = predictions - y
            dw = (1 / m) * np.dot(X.T, errors)
            db = (1 / m) * np.sum(errors)
            w = w - learning_rate * dw
            b = b - learning_rate * db
            if i % 10000 == 0:
                cost = self.compute_cost(X, y)
                logger.info(f"Iterasi {i}, Cost: {cost:.10f}")
        return w, b

    def train_model(self):
        logger.info("Starting training...")
        self.w, self.b = self.gradient_descent(self.X_norm, self.y_norm, self.w, self.b)
        logger.info(f"Training completed. Final weights: {[round(w, 7) for w in self.w]}, bias: {round(self.b, 7)}")

    def evaluate_model(self):
        logger.info("Evaluating model...")
        m = self.X_norm.shape[0]
        y_pred_norm = np.dot(self.X_norm, self.w) + self.b
        y_pred = y_pred_norm * (self.y_max - self.y_min) + self.y_min  # Denormalisasi prediksi
        y_actual = self.y_actual  # Nilai aktual dalam skala asli

        # MSE
        mse = np.mean((y_pred - y_actual) ** 2)

        # RMSE
        rmse = np.sqrt(mse)

        # RME
        rme = 0
        for i in range(m):
            if y_actual[i] != 0:  # Hindari pembagian dengan nol
                rme += abs((y_pred[i] - y_actual[i]) / y_actual[i])
            else:
                rme += abs(y_pred[i] - y_actual[i])  # Gunakan error absolut jika y=0
        rme = rme / m

        # R²
        y_mean = np.mean(y_actual)
        ss_tot = np.sum((y_actual - y_mean) ** 2)
        ss_res = np.sum((y_actual - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        logger.info(f"Evaluation Metrics:")
        logger.info(f"MSE: {mse:.7f}")
        logger.info(f"RMSE: {rmse:.7f}")
        logger.info(f"RME: {rme:.7f}")
        logger.info(f"R²: {r2:.7f}")

        return mse, rmse, rme, r2

    def predict(self, data):
        logger.info(f"Input: {data}")
        required_keys = ['GrainTypeId', 'GrainMoisture', 'GrainTemperature', 
                        'RoomTemperature', 'BurningTemperature', 'StirrerStatus', 
                        'Weight', 'TargetMoisture']

        if not all(key in data for key in required_keys):
            logger.error("Missing input fields")
            return 0.0
        if data['GrainMoisture'] <= data['TargetMoisture']:
            logger.info("Moisture already reached target.")
            return 0.0

        X_new = np.array([
            data['GrainTypeId'],
            data['GrainMoisture'],
            data['GrainTemperature'],
            data['RoomTemperature'],
            data['BurningTemperature'],
            data['StirrerStatus'],
            data['Weight']
        ])

        range_val = self.X_max - self.X_min
        range_val[range_val == 0] = 1.0
        X_new_norm = (X_new - self.X_min) / range_val
        logger.info(f"Normalized input: {X_new_norm}")

        y_pred_norm = np.dot(X_new_norm, self.w) + self.b
        logger.info(f"Normalized prediction: {y_pred_norm:.7f}")

        y_pred = y_pred_norm * (self.y_max - self.y_min) + self.y_min
        return max(0.0, y_pred)  # Pastikan prediksi tidak negatif