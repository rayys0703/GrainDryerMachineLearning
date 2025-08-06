import numpy as np
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionModel:
    TARGET_MOISTURE = 14.0

    def __init__(self, api_url="http://127.0.0.1:3333/api/dataset"):
        self.weights = None
        self.bias = 0.0
        self.training_data = []
        self.input_means = []
        self.input_stds = []
        self.output_mean = 0.0
        self.output_std = 1.0
        self.load_training_data(api_url)
        self.normalize_training_data()
        self.initialize_weights()
        self.train_model()

    def load_training_data(self, api_url):
        try:
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            raw_data = response.json()

            logger.info(f"{len(raw_data)} dataset drying process from API")

            for process in raw_data:
                grain_type_id = float(process['grain_type_id'])
                weight = float(process['berat_gabah'])
                intervals = process['intervals']

                for interval in intervals:
                    measurements = interval['sensor_data']
                    # Ambil data, jika None maka 0.0
                    drying_time = float(interval['estimasi_durasi']) if interval['estimasi_durasi'] is not None else 0.0

                    grain_temperatures = [float(m['suhu_gabah']) for m in measurements if m['suhu_gabah'] is not None]
                    grain_moistures = [float(m['kadar_air_gabah']) for m in measurements if m['kadar_air_gabah'] is not None]
                    room_temperatures = [float(m['suhu_ruangan']) for m in measurements if m['suhu_ruangan'] is not None]
                    burning_temperatures = [float(m['suhu_pembakaran']) for m in measurements if m['suhu_pembakaran'] is not None]
                    stirrer_statuses = [1 if m['status_pengaduk'] else 0 for m in measurements if m['status_pengaduk'] is not None]

                    if not (grain_temperatures and grain_moistures and room_temperatures and burning_temperatures and stirrer_statuses):
                        logger.warning(f"No valid data in interval {interval['interval_id']} (Process ID: {process['process_id']}, DryingTime: {drying_time})")
                        continue

                    avg_data = { # Data rata-rata
                        "GrainTypeId": grain_type_id,
                        "GrainTemperature": round(np.mean(grain_temperatures), 1),
                        "GrainMoisture": round(np.mean(grain_moistures), 1),
                        "RoomTemperature": round(np.mean(room_temperatures), 1),
                        "Weight": weight,
                        "BurningTemperature": burning_temperatures[0],
                        "StirrerStatus": stirrer_statuses[0],
                        "DryingTime": drying_time
                    }
                    self.training_data.append(avg_data)

            logger.info(f"Loaded {len(self.training_data)} training intervals from API")
        except requests.RequestException as e:
            logger.error(f"Error loading training data from API: {e}")
            raise
        except (ValueError, KeyError) as e:
            logger.error(f"Error processing API data: {e}")
            raise

    def normalize_training_data(self):
        features = ['GrainTypeId', 'GrainTemperature', 'GrainMoisture', 'RoomTemperature', 'Weight', 'BurningTemperature', 'StirrerStatus']
        self.input_means = [np.mean([d[key] for d in self.training_data]) for key in features]
        self.input_stds = [np.std([d[key] for d in self.training_data]) or 1.0 for key in features]
        drying_times = [d['DryingTime'] for d in self.training_data]
        self.output_mean = np.mean(drying_times)
        self.output_std = np.std(drying_times) or 1.0

        for d in self.training_data:
            d['GrainTypeId'] = (d['GrainTypeId'] - self.input_means[0]) / self.input_stds[0]
            d['GrainTemperature'] = (d['GrainTemperature'] - self.input_means[1]) / self.input_stds[1]
            d['GrainMoisture'] = (d['GrainMoisture'] - self.input_means[2]) / self.input_stds[2]
            d['RoomTemperature'] = (d['RoomTemperature'] - self.input_means[3]) / self.input_stds[3]
            d['Weight'] = (d['Weight'] - self.input_means[4]) / self.input_stds[4]
            d['BurningTemperature'] = (d['BurningTemperature'] - self.input_means[5]) / self.input_stds[5]
            d['StirrerStatus'] = (d['StirrerStatus'] - self.input_means[6]) / self.input_stds[6]
            d['DryingTime'] = (d['DryingTime'] - self.output_mean) / self.output_std

        # logger.info(f"Input means: {[round(m, 1) for m in self.input_means]}")
        # logger.info(f"Input stds: {[round(s, 1) for s in self.input_stds]}")
        # logger.info(f"Output mean: {round(self.output_mean, 1)}")
        # logger.info(f"Output std: {round(self.output_std, 1)}")

    def initialize_weights(self):
        self.weights = np.zeros(7)  # Inisialisasi weights ke 0 sesuai Jurnal 2
        # self.weights = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # Manual initialization
        self.bias = 0.0
        logger.info(f"Initialized weights: {[round(w, 3) for w in self.weights]}, bias: {self.bias}")

    def predict_linear(self, inputs):
        # Example: inputs = [1, 2, 3, 4, 5, 6, 7], weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], bias = 0.5
        # Result = (1 * 0.1) + (2 * 0.2) + (3 * 0.3) + (4 * 0.4) + (5 * 0.5) + (6 * 0.6) + (7 * 0.7) + 0.5 = 13.5
        return np.dot(inputs, self.weights) + self.bias

    def compute_cost(self, X, y):
        # X: matriks fitur input (m x n), sudah dinormalisasi
        # y: array target output, juga sudah dinormalisasi
        m = len(y) # Jumlah total dataset
        predictions = self.predict_linear(X) # Prediksi model menggunakan bobot dan bias
        errors = predictions - y # Selisih prediksi dan target
        cost = (1 / (2 * m)) * np.sum(errors ** 2)  # MSE sesuai Jurnal 1, # MSE dibagi 2
        return cost

    def compute_gradients(self, X, y):
        m = len(y)
        predictions = self.predict_linear(X)
        errors = predictions - y
        grad_weights = (1 / m) * np.dot(X.T, errors)  # Gradient untuk weights sesuai Jurnal 1
        grad_bias = (1 / m) * np.sum(errors)  # Gradient untuk bias
        return grad_weights, grad_bias

    def adaptive_learning_rate(self, epoch, initial_lr=0.01, decay_rate=0.1, decay_steps=100):
        # Learning rate decay sesuai Jurnal 1
        return initial_lr / (1 + decay_rate * (epoch // decay_steps))

    def gradient_descent(self, X, y, initial_lr=0.01, epochs=10000, momentum=0.9):
        m = len(y)
        velocity_weights = np.zeros_like(self.weights)  # Momentum untuk weights
        velocity_bias = 0.0  # Momentum untuk bias

        for epoch in range(epochs):
            grad_weights, grad_bias = self.compute_gradients(X, y)

            lr = self.adaptive_learning_rate(epoch, initial_lr) # Learning rate adaptif

            # Update bobot dan bias dengan momentum sesuai jurnal 1
            velocity_weights = momentum * velocity_weights - lr * grad_weights
            velocity_bias = momentum * velocity_bias - lr * grad_bias

            # Update weights dan bias
            self.weights += velocity_weights
            # self.weights = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) // Manual initialization

            self.bias += velocity_bias
            # self.bias = 0.0 # Manual initialization

            cost = self.compute_cost(X, y)
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}: Cost = {cost:.4f}, Learning Rate = {lr:.6f}, Weights = {[round(w, 3) for w in self.weights]}, Bias = {round(self.bias, 3)}")

        return self.weights, self.bias

    def train_model(self):
        X = np.array([
            [d['GrainTypeId'], d['GrainTemperature'], d['GrainMoisture'], d['RoomTemperature'], d['Weight'], d['BurningTemperature'], d['StirrerStatus']]
            for d in self.training_data
        ])
        y = np.array([d['DryingTime'] for d in self.training_data])
        logger.info("Starting model training LR...")
        self.gradient_descent(X, y)
        mse = self.calculate_mse(X, y)
        r2 = self.calculate_r2(X, y)
        logger.info(f"Training OK. Final MSE on training data: {mse:.4f}, R2: {r2:.4f}")

    def calculate_mse(self, X, y_actual):
        m = len(y_actual)
        y_predicted = self.predict_linear(X)
        mse = np.mean((y_actual - y_predicted) ** 2)
        return mse

    def calculate_r2(self, X, y_actual): # R Square sesuai Jurnal 2
        y_predicted = self.predict_linear(X)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        ss_res = np.sum((y_actual - y_predicted) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

    def predict(self, data):
        logger.info(f"Input data: {{'GrainTypeId': {data['GrainTypeId']}, 'GrainTemperature': {data['GrainTemperature']:.1f}, 'GrainMoisture': {data['GrainMoisture']:.1f}, 'RoomTemperature': {data['RoomTemperature']:.1f}, 'Weight': {data['Weight']:.1f}, 'BurningTemperature': {data['BurningTemperature']:.1f}, 'StirrerStatus': {data['StirrerStatus']}}}")
        required_keys = ['GrainTypeId', 'GrainTemperature', 'GrainMoisture', 'RoomTemperature', 'Weight', 'BurningTemperature', 'StirrerStatus']
        if not all(key in data for key in required_keys) or not all(isinstance(data[key], (int, float)) for key in required_keys) or any(data[key] < 0 for key in required_keys if key != 'StirrerStatus'):
            logger.error("Invalid input data detected")
            return 0.0
        if data['GrainMoisture'] <= self.TARGET_MOISTURE:
            logger.info("Moisture target reached (14% or less), drying complete")
            return 0.0
        normalized_data = np.array([
            (data['GrainTypeId'] - self.input_means[0]) / self.input_stds[0],
            (data['GrainTemperature'] - self.input_means[1]) / self.input_stds[1],
            (data['GrainMoisture'] - self.input_means[2]) / self.input_stds[2],
            (data['RoomTemperature'] - self.input_means[3]) / self.input_stds[3],
            (data['Weight'] - self.input_means[4]) / self.input_stds[4],
            (data['BurningTemperature'] - self.input_means[5]) / self.input_stds[5],
            (data['StirrerStatus'] - self.input_means[6]) / self.input_stds[6]
        ])
        normalized_result = self.predict_linear(normalized_data)
        result = normalized_result * self.output_std + self.output_mean
        result = max(result, 0)
        result = min(result, 2880)
        mse = self.calculate_mse(np.array([
            [d['GrainTypeId'], d['GrainTemperature'], d['GrainMoisture'], d['RoomTemperature'], d['Weight'], d['BurningTemperature'], d['StirrerStatus']]
            for d in self.training_data
        ]), np.array([d['DryingTime'] for d in self.training_data]))
        r2 = self.calculate_r2(np.array([
            [d['GrainTypeId'], d['GrainTemperature'], d['GrainMoisture'], d['RoomTemperature'], d['Weight'], d['BurningTemperature'], d['StirrerStatus']]
            for d in self.training_data
        ]), np.array([d['DryingTime'] for d in self.training_data]))
        logger.info(f"Prediction successful: {result:.2f} minutes, MSE on training data: {mse:.4f}, R2: {r2:.4f}")
        return result if not np.isnan(result) else 0.0