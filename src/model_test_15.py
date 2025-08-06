import numpy as np
import logging

# Logging untuk mencatat info saat training dan debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionModel:
    def __init__(self):
        # Inisialisasi bobot dan bias
        self.w = None
        self.b = None
        self.X_min = None
        self.X_max = None
        self.y_min = None
        self.y_max = None
        self.training_data = []

        # Ambil data dan siapkan training
        self.load_training_data()
        self.normalize_training_data()
        self.initialize_weights()
        self.train_model()
        self.evaluate_model()

    def load_training_data(self):
        self.training_data = [
            {"GrainTypeId": 1.0, "GrainMoisture": 28.0000000, "GrainTemperature": 26.0000000, "RoomTemperature": 28.0000000, "BurningTemperature": 300.9934283, "Weight": 20000.0000000, "StirrerStatus": 1.0, "DryingTime": 1200.0000000},
            {"GrainTypeId": 1.0, "GrainMoisture": 27.9990278, "GrainTemperature": 26.0009722, "RoomTemperature": 28.0349048, "BurningTemperature": 300.0724664, "Weight": 20000.0000000, "StirrerStatus": 1.0, "DryingTime": 1199.9166667},
            {"GrainTypeId": 1.0, "GrainMoisture": 27.9980556, "GrainTemperature": 26.0019444, "RoomTemperature": 28.0697990, "BurningTemperature": 301.9929418, "Weight": 20000.0000000, "StirrerStatus": 1.0, "DryingTime": 1199.8333333},
            {"GrainTypeId": 1.0, "GrainMoisture": 27.9970833, "GrainTemperature": 26.0029167, "RoomTemperature": 28.1046719, "BurningTemperature": 304.0913443, "Weight": 20000.0000000, "StirrerStatus": 1.0, "DryingTime": 1199.7500000},
            {"GrainTypeId": 1.0, "GrainMoisture": 27.9961111, "GrainTemperature": 26.0038889, "RoomTemperature": 28.1395129, "BurningTemperature": 300.9234243, "Weight": 20000.0000000, "StirrerStatus": 1.0, "DryingTime": 1199.6666667},
            {"GrainTypeId": 1.0, "GrainMoisture": 27.9951389, "GrainTemperature": 26.0048611, "RoomTemperature": 28.1743115, "BurningTemperature": 301.2682079, "Weight": 20000.0000000, "StirrerStatus": 1.0, "DryingTime": 1199.5833333},
            {"GrainTypeId": 1.0, "GrainMoisture": 27.9941667, "GrainTemperature": 26.0058333, "RoomTemperature": 28.2090569, "BurningTemperature": 305.2375425, "Weight": 20000.0000000, "StirrerStatus": 1.0, "DryingTime": 1199.5000000},
            {"GrainTypeId": 1.0, "GrainMoisture": 27.9931944, "GrainTemperature": 26.0068056, "RoomTemperature": 28.2437387, "BurningTemperature": 303.9540884, "Weight": 20000.0000000, "StirrerStatus": 1.0, "DryingTime": 1199.4166667},
            {"GrainTypeId": 1.0, "GrainMoisture": 27.9922222, "GrainTemperature": 26.0077778, "RoomTemperature": 28.2783462, "BurningTemperature": 301.8174248, "Weight": 20000.0000000, "StirrerStatus": 1.0, "DryingTime": 1199.3333333},
            {"GrainTypeId": 1.0, "GrainMoisture": 27.9912500, "GrainTemperature": 26.0087500, "RoomTemperature": 28.3128689, "BurningTemperature": 304.1752900, "Weight": 20000.0000000, "StirrerStatus": 1.0, "DryingTime": 1199.2500000},
            {"GrainTypeId": 1.0, "GrainMoisture": 27.9902778, "GrainTemperature": 26.0097222, "RoomTemperature": 28.3472964, "BurningTemperature": 302.4933660, "Weight": 20000.0000000, "StirrerStatus": 1.0, "DryingTime": 1199.1666667},
            {"GrainTypeId": 1.0, "GrainMoisture": 27.9893056, "GrainTemperature": 26.0106944, "RoomTemperature": 28.3816180, "BurningTemperature": 302.8146064, "Weight": 20000.0000000, "StirrerStatus": 1.0, "DryingTime": 1199.0833333},
            {"GrainTypeId": 1.0, "GrainMoisture": 27.9883333, "GrainTemperature": 26.0116667, "RoomTemperature": 28.4158234, "BurningTemperature": 304.5512910, "Weight": 20000.0000000, "StirrerStatus": 1.0, "DryingTime": 1199.0000000},
            {"GrainTypeId": 1.0, "GrainMoisture": 27.9873611, "GrainTemperature": 26.0126389, "RoomTemperature": 28.4499021, "BurningTemperature": 300.5571510, "Weight": 20000.0000000, "StirrerStatus": 1.0, "DryingTime": 1198.9166667},
            {"GrainTypeId": 1.0, "GrainMoisture": 27.9863889, "GrainTemperature": 26.0136111, "RoomTemperature": 28.4838438, "BurningTemperature": 301.2448800, "Weight": 20000.0000000, "StirrerStatus": 1.0, "DryingTime": 1198.8333333},
        ]
        logger.info(f"Loaded {len(self.training_data)} training intervals from static data")

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

    def compute_cost(self, X, y, w, b):
        m = X.shape[0]
        cost_sum = 0
        for i in range(m):
            f_wb = np.dot(w, X[i]) + b
            cost = (f_wb - y[i]) ** 2
            cost_sum += cost
        total_cost = (1 / (2 * m)) * cost_sum
        return total_cost

    def compute_gradient(self, X, y, w, b):
        m = X.shape[0]
        dj_dw = np.zeros(w.shape)
        dj_db = 0

        for i in range(m):
            f_wb = np.dot(w, X[i]) + b
            dj_dw_i = (f_wb - y[i]) * X[i]
            dj_db_i = f_wb - y[i]
            dj_dw += dj_dw_i
            dj_db += dj_db_i

        dj_dw = dj_dw / m
        dj_db = dj_db / m
        return dj_dw, dj_db

    def gradient_descent(self, X, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
        J_history = []
        p_history = []
        b = b_in
        w = w_in

        for i in range(num_iters):
            dj_dw, dj_db = gradient_function(X, y, w, b)
            w = w - alpha * dj_dw
            b = b - alpha * dj_db

            if i < 10000:
                J_history.append(cost_function(X, y, w, b))
                p_history.append([w, b])

            if i % max(1, num_iters // 10) == 0:
                logger.info(f"Iteration {i:4d}: Cost {J_history[-1]:0.2e}   dw: {dj_dw[0]:0.3e}   db: {dj_db:0.3e}  w[0]: {w[0]:0.3e}   b: {b:0.3e}")

        return w, b, J_history, p_history

    def train_model(self):
        logger.info("Starting training...")
        iterations = 100000
        alpha = 0.01
        self.w, self.b, J_history, p_history = self.gradient_descent(
            self.X_norm, self.y_norm, self.w, self.b, alpha, iterations, 
            self.compute_cost, self.compute_gradient
        )
        logger.info(f"Training completed. Final weights: {[round(w, 7) for w in self.w]}, bias: {round(self.b, 7)}")
        logger.info(f"Final cost: {J_history[-1]:.2e}")

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