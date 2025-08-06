import paho.mqtt.client as mqtt
import json
import logging
import requests
import time
import threading
from flask import Flask, request, jsonify
from model import PredictionModel
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class MQTTService:
    def __init__(self, model, broker="broker.hivemq.com", port=1883, 
                 input_topic="iot/sensor/datagabah2", output_topic="iot/prediction/dryingtime",
                 laravel_api_url="http://127.0.0.1:3333/api/prediction/receive",
                 username="graindryer", password="polindra"):
        self.model = model
        self.broker = broker
        self.port = port
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.laravel_api_url = laravel_api_url
        self.username = username
        self.password = password
        self.client = mqtt.Client()
        self.prediction_trigger = None
        self.lock = threading.Lock()

        # Setup HTTP session with retries
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info(f"Terhubung ke broker MQTT: {self.broker}:{self.port}")
            self.client.subscribe(self.input_topic, qos=0)
            logger.info(f"Berlangganan topik: {self.input_topic}")

            # Cek status pengeringan yang aktif melalui API Laravel
            try:
                response = self.session.get("http://127.0.0.1:3333/api/realtime-data", timeout=10)
                response.raise_for_status()
                data = response.json()
                drying_process = data.get("drying_process")
                if drying_process is not None and drying_process.get("status") == "ongoing":
                    process_id = drying_process["process_id"]
                    berat_gabah = drying_process["berat_gabah"]
                    kadar_air_target = drying_process["kadar_air_target"]
                    grain_type_id = drying_process["grain_type_id"]

                    # Inisiasi prediction_trigger jika pengeringan aktif
                    with self.lock:
                        self.prediction_trigger = {
                            "process_id": process_id,
                            "grain_type_id": grain_type_id,
                            "berat_gabah": berat_gabah,
                            "kadar_air_target": kadar_air_target
                        }
                        logger.info(f"Pengeringan aktif ditemukan (Process ID: {process_id}). Prediksi otomatis dimulai.")
                else:
                    logger.info("Tidak ada pengeringan aktif.")
            except requests.RequestException as e:
                logger.error(f"Gagal memeriksa status pengeringan: {e}")
        else:
            logger.error(f"Gagal terhubung ke broker: kode {rc}")
            raise Exception(f"Koneksi MQTT gagal: kode {rc}")

    def on_message(self, client, userdata, msg):
        try:
            with self.lock:
                if not self.prediction_trigger:
                    logger.info("Data sensor diabaikan: Tidak ada prediksi aktif")
                    return

                payload = json.loads(msg.payload.decode())
                if not (payload.get("points") and isinstance(payload["points"], list) and payload.get("room_temperature")):
                    logger.error("Format data sensor tidak valid")
                    return

                points = payload["points"]
                grain_temps = [p["grain_temperature"] for p in points if "grain_temperature" in p]
                grain_moists = [p["grain_moisture"] for p in points if "grain_moisture" in p]

                if not (grain_temps and grain_moists):
                    logger.error("Data suhu atau kadar air gabah tidak ada")
                    return

                avg_temp = round(sum(grain_temps) / len(grain_temps), 1)
                avg_moist = round(sum(grain_moists) / len(grain_moists), 1)
                room_temp = payload["room_temperature"]
                weight = self.prediction_trigger.get("berat_gabah", 0)

                if weight < 100:
                    logger.error(f"Berat gabah tidak valid: {weight} kg")
                    return

                process_id = self.prediction_trigger.get("process_id")
                kadar_air_target = self.prediction_trigger.get("kadar_air_target", 0)

                laravel_payload = {
                    "process_id": process_id,
                    "points": points,
                    "room_temperature": room_temp,
                    "avg_grain_temperature": avg_temp,
                    "avg_grain_moisture": avg_moist,
                    "predicted_drying_time": 0.0,
                    "weight": weight,
                    "timestamp": int(time.time())
                }

                # if avg_moist <= kadar_air_target:
                #     logger.info(f"Target kadar air tercapai: {avg_moist}% <= {kadar_air_target}% (ID: {process_id})")
                #     laravel_payload["predicted_drying_time"] = 0.0
                #     try:
                #         response = self.session.post(self.laravel_api_url, json=laravel_payload, timeout=15)
                #         response.raise_for_status()
                #         logger.info(f"Data terkirim ke Laravel (ID: {process_id}, Kadar Air Tercapai)")
                #         self.prediction_trigger = None  # Clear trigger after successful send
                #     except requests.RequestException as e:
                #         logger.error(f"Gagal kirim data ke Laravel: {e} (ID: {process_id})")
                #     return

                prediction_input = {
                    "GrainTemperature": avg_temp,
                    "GrainMoisture": avg_moist,
                    "RoomTemperature": room_temp,
                    "Weight": weight
                }

                predicted_time = self.model.predict(prediction_input)
                logger.info(f"Prediksi durasi pengeringan: {predicted_time:.2f} menit (ID: {process_id})")
                laravel_payload["predicted_drying_time"] = predicted_time

                try:
                    response = self.session.post(self.laravel_api_url, json=laravel_payload, timeout=15)
                    response.raise_for_status()
                    logger.info(f"Data terkirim ke Laravel (ID: {process_id}, Estimasi Durasi: {predicted_time:.2f} menit)")
                except requests.RequestException as e:
                    logger.error(f"Gagal kirim data ke Laravel: {e} (ID: {process_id})")

                # prediction_msg = json.dumps({"process_id": process_id, "DryingTime": predicted_time})
                # self.client.publish(self.output_topic, prediction_msg, qos=0)
                # logger.info(f"Prediksi diterbitkan ke {self.output_topic} (ID: {process_id})")

        except Exception as e:
            logger.error(f"Error memproses pesan sensor: {e} (ID: {self.prediction_trigger.get('process_id') if self.prediction_trigger else 'N/A'})")

    def start_mqtt(self):
        try:
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()
            logger.info("Layanan MQTT dimulai")
        except Exception as e:
            logger.error(f"Gagal memulai MQTT: {e}")
            raise

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("Layanan MQTT dihentikan")

@app.route('/predict', methods=['POST'])
def trigger_prediction():
    try:
        data = request.get_json()
        required_keys = ['process_id', 'grain_type_id', 'berat_gabah', 'kadar_air_target']
        if not all(key in data for key in required_keys):
            logger.error("Data prediksi tidak lengkap")
            return jsonify({"error": "Data tidak lengkap"}), 400

        if not isinstance(data['berat_gabah'], (int, float)) or data['berat_gabah'] < 100:
            logger.error(f"Berat gabah tidak valid: {data['berat_gabah']} kg")
            return jsonify({"error": "Berat gabah minimal 100 kg"}), 400

        if not isinstance(data['kadar_air_target'], (int, float)) or data['kadar_air_target'] < 0 or data['kadar_air_target'] > 100:
            logger.error(f"Target kadar air tidak valid: {data['kadar_air_target']}%")
            return jsonify({"error": "Target kadar air harus antara 0-100"}), 400

        with app.mqtt_service.lock:
            if app.mqtt_service.prediction_trigger:
                logger.warning(f"Prediksi sudah aktif (ID: {app.mqtt_service.prediction_trigger['process_id']})")
                return jsonify({"error": "Prediksi sudah aktif"}), 409
            app.mqtt_service.prediction_trigger = data
            logger.info(f"Mulai prediksi: ID {data['process_id']}, berat_gabah: {data['berat_gabah']} kg, target_kadar_air: {data['kadar_air_target']}%")

        return jsonify({"message": "Prediksi dimulai, menunggu data sensor"}), 200
    except Exception as e:
        logger.error(f"Gagal memulai prediksi: {e}")
        return jsonify({"error": "Gagal memulai prediksi"}), 500

@app.route('/stop-prediction', methods=['POST'])
def stop_prediction():
    try:
        data = request.get_json()
        if 'process_id' not in data:
            logger.error("ID proses tidak ada")
            return jsonify({"error": "ID proses tidak ada"}), 400

        with app.mqtt_service.lock:
            if not app.mqtt_service.prediction_trigger:
                logger.info("Tidak ada prediksi aktif untuk dihentikan")
                return jsonify({"message": "Tidak ada prediksi aktif"}), 200
            if app.mqtt_service.prediction_trigger['process_id'] != data['process_id']:
                logger.error(f"ID proses tidak cocok: {data['process_id']} != {app.mqtt_service.prediction_trigger['process_id']}")
                return jsonify({"error": "ID proses tidak cocok"}), 400
            process_id = app.mqtt_service.prediction_trigger['process_id']
            app.mqtt_service.prediction_trigger = None
            logger.info(f"Prediksi dihentikan: ID {process_id}")

        return jsonify({"message": "Prediksi dihentikan"}), 200
    except Exception as e:
        logger.error(f"Gagal menghentikan prediksi: {e}")
        return jsonify({"error": "Gagal menghentikan prediksi"}), 500

def run_flask(mqtt_service):
    app.mqtt_service = mqtt_service
    app.run(host='0.0.0.0', port=5000, debug=False)

# import logging
# import threading
# from model import PredictionModel
# from mqtt_service import MQTTService, run_flask

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def main():
#     mqtt_service = None
#     try:
#         model = PredictionModel()
#         logger.info("Prediction model initialized")

#         mqtt_service = MQTTService(
#             model=model,
#             broker="broker.hivemq.com",
#             port=1883,
#             input_topic="iot/sensor/datagabah2",
#             output_topic="iot/prediction/dryingtime",
#             laravel_api_url="http://127.0.0.1:3333/api/prediction/receive",
#             username="graindryer",
#             password="polindra"
#         )
#         logger.info("MQTT service initialized")

#         mqtt_thread = threading.Thread(target=mqtt_service.start_mqtt, daemon=True)
#         mqtt_thread.start()
#         logger.info("MQTT thread started")

#         logger.info("Starting Flask server on http://127.0.0.1:5000")
#         run_flask(mqtt_service)

#     except KeyboardInterrupt:
#         logger.info("Shutting down...")
#         if mqtt_service:
#             mqtt_service.stop()
#     except Exception as e:
#         logger.error(f"Error in main: {e}")
#         if mqtt_service:
#             mqtt_service.stop()

# if __name__ == "__main__":
#     main()


# #src\model.py
# import json
# import numpy as np
# import logging
# import os

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class PredictionModel:
#     TARGET_MOISTURE = 14.0  # Target kadar air 14%

#     def __init__(self, data_path=os.path.join(os.path.dirname(__file__), "..", "data", "training-data.json")):
#         self.weights = None  # Bobot: [w1, w2, w3, w4]
#         self.bias = 0.0      # Bias: b
#         self.training_data = []
#         self.input_means = []
#         self.input_stds = []
#         self.output_mean = 0.0
#         self.output_std = 1.0
#         self.load_training_data(data_path)
#         self.normalize_training_data()
#         self.initialize_weights()
#         self.train_model()

#     def load_training_data(self, data_path):
#         """Load training data from JSON file."""
#         try:
#             with open(data_path, 'r') as f:
#                 raw_data = json.load(f)
#             for group in raw_data:
#                 measurements = group[0]
#                 drying_time = group[1]["DryingTime"]
#                 avg_data = {
#                     "GrainTemperature": round(np.mean([m["GrainTemperature"] for m in measurements]), 1),
#                     "GrainMoisture": round(np.mean([m["GrainMoisture"] for m in measurements]), 1),
#                     "RoomTemperature": round(np.mean([m["RoomTemperature"] for m in measurements]), 1),
#                     "Weight": round(np.mean([m["Weight"] for m in measurements]), 1),
#                     "DryingTime": drying_time
#                 }
#                 self.training_data.append(avg_data)
#             logger.info(f"Loaded {len(self.training_data)} training groups")
#         except Exception as e:
#             logger.error(f"Error loading training data: {e}")
#             raise

#     def normalize_training_data(self):
#         """Normalize training data using z-score."""
#         features = ['GrainTemperature', 'GrainMoisture', 'RoomTemperature', 'Weight']
#         self.input_means = [np.mean([d[key] for d in self.training_data]) for key in features]
#         self.input_stds = [np.std([d[key] for d in self.training_data]) or 1.0 for key in features]
#         drying_times = [d['DryingTime'] for d in self.training_data]
#         self.output_mean = np.mean(drying_times)
#         self.output_std = np.std(drying_times) or 1.0

#         for d in self.training_data:
#             d['GrainTemperature'] = (d['GrainTemperature'] - self.input_means[0]) / self.input_stds[0]
#             d['GrainMoisture'] = (d['GrainMoisture'] - self.input_means[1]) / self.input_stds[1]
#             d['RoomTemperature'] = (d['RoomTemperature'] - self.input_means[2]) / self.input_stds[2]
#             d['Weight'] = (d['Weight'] - self.input_means[3]) / self.input_stds[3]
#             d['DryingTime'] = (d['DryingTime'] - self.output_mean) / self.output_std

#         logger.info(f"Input means: {[round(m, 1) for m in self.input_means]}")
#         logger.info(f"Input stds: {[round(s, 1) for s in self.input_stds]}")
#         logger.info(f"Output mean: {round(self.output_mean, 1)}")
#         logger.info(f"Output std: {round(self.output_std, 1)}")

#     def initialize_weights(self):
#         """Initialize weights with meaningful starting values."""
#         self.weights = np.array([0.01, 0.2, 0.01, 0.05])  # w2 lebih besar, w4 memengaruhi Weight
#         self.bias = 0.0
#         logger.info(f"Initialized weights: {[round(w, 3) for w in self.weights]}, bias: {self.bias}")

#     def predict_linear(self, inputs):
#         """Compute prediction: w1*x1 + w2*x2 + w3*x3 + w4*x4 + b."""
#         return np.dot(inputs, self.weights) + self.bias

#     def compute_cost(self, X, y):
#         """Compute Mean Squared Error (MSE) cost function."""
#         m = len(y)
#         predictions = self.predict_linear(X)
#         errors = predictions - y
#         cost = (1 / (2 * m)) * np.sum(errors ** 2)
#         return cost

#     def gradient_descent(self, X, y, learning_rate=0.01, epochs=1000):
#         """Perform gradient descent to optimize weights and bias."""
#         m = len(y)
#         for epoch in range(epochs):
#             predictions = self.predict_linear(X)
#             errors = predictions - y
#             grad_weights = (1 / m) * np.dot(X.T, errors)
#             grad_bias = (1 / m) * np.sum(errors)
#             self.weights -= learning_rate * grad_weights
#             self.bias -= learning_rate * grad_bias
#             cost = self.compute_cost(X, y)
#             if epoch % 50 == 0:
#                 logger.info(f"Epoch {epoch}: Cost = {cost:.4f}, Weights = {[round(w, 3) for w in self.weights]}, Bias = {round(self.bias, 3)}")

#     def train_model(self):
#         """Train the model using gradient descent."""
#         X = np.array([
#             [d['GrainTemperature'], d['GrainMoisture'], d['RoomTemperature'], d['Weight']]
#             for d in self.training_data
#         ])
#         y = np.array([d['DryingTime'] for d in self.training_data])
#         logger.info("Starting model training with gradient descent...")
#         self.gradient_descent(X, y)
#         logger.info("Training completed")

#     def predict(self, data):
#         """Predict drying time based on input data."""
#         logger.info(f"Input data: {{'GrainTemperature': {data['GrainTemperature']:.1f}, 'GrainMoisture': {data['GrainMoisture']:.1f}, 'RoomTemperature': {data['RoomTemperature']:.1f}, 'Weight': {data['Weight']:.1f}}}")
#         required_keys = ['GrainTemperature', 'GrainMoisture', 'RoomTemperature', 'Weight']
#         if not all(key in data for key in required_keys) or not all(isinstance(data[key], (int, float)) for key in required_keys) or any(data[key] < 0 for key in required_keys):
#             logger.error("Invalid input data detected")
#             return 0.0
#         if data['GrainMoisture'] <= self.TARGET_MOISTURE:
#             logger.info("Moisture target reached (14% or less), drying complete")
#             return 0.0
#         max_grain_temp = max(d['GrainTemperature'] * self.input_stds[0] + self.input_means[0] for d in self.training_data)
#         min_grain_temp = min(d['GrainTemperature'] * self.input_stds[0] + self.input_means[0] for d in self.training_data)
#         max_weight = max(d['Weight'] * self.input_stds[3] + self.input_means[3] for d in self.training_data)
#         min_weight = min(d['Weight'] * self.input_stds[3] + self.input_means[3] for d in self.training_data)
#         if data['GrainTemperature'] > max_grain_temp or data['GrainTemperature'] < min_grain_temp:
#             logger.warning(f"GrainTemperature ({data['GrainTemperature']:.1f}°C) is outside training data range ({min_grain_temp:.1f}–{max_grain_temp:.1f}°C).")
#         if data['Weight'] > max_weight or data['Weight'] < min_weight:
#             logger.warning(f"Weight ({data['Weight']:.1f} tons) is outside training data range ({min_weight:.1f}–{max_weight:.1f} tons).")
#         normalized_data = np.array([
#             (data['GrainTemperature'] - self.input_means[0]) / self.input_stds[0],
#             (data['GrainMoisture'] - self.input_means[1]) / self.input_stds[1],
#             (data['RoomTemperature'] - self.input_means[2]) / self.input_stds[2],
#             (data['Weight'] - self.input_means[3]) / self.input_stds[3]
#         ])
#         normalized_result = self.predict_linear(normalized_data)
#         result = normalized_result * self.output_std + self.output_mean
#         result = max(result, 0)
#         result = min(result, 720)
#         logger.info(f"Normalized prediction: {normalized_result:.4f}")
#         logger.info(f"Denormalized prediction: {result:.2f} minutes")
#         return result if not np.isnan(result) else 0.0
    
# # if __name__ == "__main__":
# #     model = PredictionModel()
# #     test_data = {"GrainTemperature": 40.0, "GrainMoisture": 15.0, "RoomTemperature": 32.0, "Weight": 2.0}
# #     predicted_time = model.predict(test_data)
# #     print(f"Predicted drying time: {predicted_time:.2f} minutes")

############

# import numpy as np
# import logging
# import requests

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class PredictionModel:
#     TARGET_MOISTURE = 14.0  # Target kadar air 14%

#     def __init__(self, api_url="http://127.0.0.1:3333/api/training-data"):
#         self.weights = None  # Bobot: [w1, w2, w3, w4]
#         self.bias = 0.0      # Bias: b
#         self.training_data = []
#         self.input_means = []
#         self.input_stds = []
#         self.output_mean = 0.0
#         self.output_std = 1.0
#         self.load_training_data(api_url)
#         self.normalize_training_data()
#         self.initialize_weights()
#         self.train_model()

#     def load_training_data(self, api_url):
#         """Load training data from Laravel API."""
#         try:
#             response = requests.get(api_url, timeout=10)
#             response.raise_for_status()
#             raw_data = response.json()

#             logger.info(f"Received {len(raw_data)} groups from API")

#             for group in raw_data:
#                 measurements = group['measurements']
#                 drying_time = group['DryingTime']

#                 # Konversi nilai ke float untuk memastikan numerik
#                 grain_temperatures = [float(m["GrainTemperature"]) for m in measurements]
#                 grain_moistures = [float(m["GrainMoisture"]) for m in measurements]
#                 room_temperatures = [float(m["RoomTemperature"]) for m in measurements]
#                 weights = [float(m["Weight"]) for m in measurements]

#                 # Log contoh data untuk debugging
#                 logger.debug(f"Group sample: GrainTemperature={grain_temperatures[:2]}, DryingTime={drying_time}")

#                 avg_data = {
#                     "GrainTemperature": round(np.mean(grain_temperatures), 1),
#                     "GrainMoisture": round(np.mean(grain_moistures), 1),
#                     "RoomTemperature": round(np.mean(room_temperatures), 1),
#                     "Weight": round(np.mean(weights), 1),
#                     "DryingTime": drying_time
#                 }
#                 self.training_data.append(avg_data)
#             logger.info(f"Loaded {len(self.training_data)} training groups from API")
#         except requests.RequestException as e:
#             logger.error(f"Error loading training data from API: {e}")
#             raise
#         except (ValueError, KeyError) as e:
#             logger.error(f"Error processing API data: {e}")
#             raise

#     def normalize_training_data(self):
#         """Normalize training data using z-score."""
#         features = ['GrainTemperature', 'GrainMoisture', 'RoomTemperature', 'Weight']
#         self.input_means = [np.mean([d[key] for d in self.training_data]) for key in features]
#         self.input_stds = [np.std([d[key] for d in self.training_data]) or 1.0 for key in features]
#         drying_times = [d['DryingTime'] for d in self.training_data]
#         self.output_mean = np.mean(drying_times)
#         self.output_std = np.std(drying_times) or 1.0

#         for d in self.training_data:
#             d['GrainTemperature'] = (d['GrainTemperature'] - self.input_means[0]) / self.input_stds[0]
#             d['GrainMoisture'] = (d['GrainMoisture'] - self.input_means[1]) / self.input_stds[1]
#             d['RoomTemperature'] = (d['RoomTemperature'] - self.input_means[2]) / self.input_stds[2]
#             d['Weight'] = (d['Weight'] - self.input_means[3]) / self.input_stds[3]
#             d['DryingTime'] = (d['DryingTime'] - self.output_mean) / self.output_std

#         logger.info(f"Input means: {[round(m, 1) for m in self.input_means]}")
#         logger.info(f"Input stds: {[round(s, 1) for s in self.input_stds]}")
#         logger.info(f"Output mean: {round(self.output_mean, 1)}")
#         logger.info(f"Output std: {round(self.output_std, 1)}")

#     def initialize_weights(self):
#         """Initialize weights with meaningful starting values."""
#         self.weights = np.array([0.01, 0.2, 0.01, 0.05])  # w2 lebih besar, w4 memengaruhi Weight
#         self.bias = 0.0
#         logger.info(f"Initialized weights: {[round(w, 3) for w in self.weights]}, bias: {self.bias}")

#     def predict_linear(self, inputs):
#         """Compute prediction: w1*x1 + w2*x2 + w3*x3 + w4*x4 + b."""
#         return np.dot(inputs, self.weights) + self.bias

#     def compute_cost(self, X, y):
#         """Compute Mean Squared Error (MSE) cost function."""
#         m = len(y)
#         predictions = self.predict_linear(X)
#         errors = predictions - y
#         cost = (1 / (2 * m)) * np.sum(errors ** 2)
#         return cost

#     def gradient_descent(self, X, y, learning_rate=0.01, epochs=1000):
#         """Perform gradient descent to optimize weights and bias."""
#         m = len(y)
#         for epoch in range(epochs):
#             predictions = self.predict_linear(X)
#             errors = predictions - y
#             grad_weights = (1 / m) * np.dot(X.T, errors)
#             grad_bias = (1 / m) * np.sum(errors)
#             self.weights -= learning_rate * grad_weights
#             self.bias -= learning_rate * grad_bias
#             cost = self.compute_cost(X, y)
#             if epoch % 50 == 0:
#                 logger.info(f"Epoch {epoch}: Cost = {cost:.4f}, Weights = {[round(w, 3) for w in self.weights]}, Bias = {round(self.bias, 3)}")

#     def train_model(self):
#         """Train the model using gradient descent."""
#         X = np.array([
#             [d['GrainTemperature'], d['GrainMoisture'], d['RoomTemperature'], d['Weight']]
#             for d in self.training_data
#         ])
#         y = np.array([d['DryingTime'] for d in self.training_data])
#         logger.info("Starting model training with gradient descent...")
#         self.gradient_descent(X, y)

#         # Calculate and log MSE after training
#         mse = self.calculate_mse(X, y)
#         logger.info(f"Training completed. Final MSE on training data: {mse:.4f}")

#     def calculate_mse(self, X, y_actual):
#         """Calculate MSE based on training data and predictions."""
#         m = len(y_actual)
#         y_predicted = self.predict_linear(X)
#         mse = np.mean((y_actual - y_predicted) ** 2)
#         return mse

#     def predict(self, data):
#         """Predict drying time based on input data."""
#         logger.info(f"Input data: {{'GrainTemperature': {data['GrainTemperature']:.1f}, 'GrainMoisture': {data['GrainMoisture']:.1f}, 'RoomTemperature': {data['RoomTemperature']:.1f}, 'Weight': {data['Weight']:.1f}}}")
#         required_keys = ['GrainTemperature', 'GrainMoisture', 'RoomTemperature', 'Weight']
#         if not all(key in data for key in required_keys) or not all(isinstance(data[key], (int, float)) for key in required_keys) or any(data[key] < 0 for key in required_keys):
#             logger.error("Invalid input data detected")
#             return 0.0
#         if data['GrainMoisture'] <= self.TARGET_MOISTURE:
#             logger.info("Moisture target reached (14% or less), drying complete")
#             return 0.0
#         max_grain_temp = max(d['GrainTemperature'] * self.input_stds[0] + self.input_means[0] for d in self.training_data)
#         min_grain_temp = min(d['GrainTemperature'] * self.input_stds[0] + self.input_means[0] for d in self.training_data)
#         max_weight = max(d['Weight'] * self.input_stds[3] + self.input_means[3] for d in self.training_data)
#         min_weight = min(d['Weight'] * self.input_stds[3] + self.input_means[3] for d in self.training_data)
#         if data['GrainTemperature'] > max_grain_temp or data['GrainTemperature'] < min_grain_temp:
#             logger.warning(f"GrainTemperature ({data['GrainTemperature']:.1f}°C) is outside training data range ({min_grain_temp:.1f}–{max_grain_temp:.1f}°C).")
#         if data['Weight'] > max_weight or data['Weight'] < min_weight:
#             logger.warning(f"Weight ({data['Weight']:.1f} kg) is outside training data range ({min_weight:.1f}–{max_weight:.1f} kg).")
#         normalized_data = np.array([
#             (data['GrainTemperature'] - self.input_means[0]) / self.input_stds[0],
#             (data['GrainMoisture'] - self.input_means[1]) / self.input_stds[1],
#             (data['RoomTemperature'] - self.input_means[2]) / self.input_stds[2],
#             (data['Weight'] - self.input_means[3]) / self.input_stds[3]
#         ])
#         normalized_result = self.predict_linear(normalized_data)
#         result = normalized_result * self.output_std + self.output_mean
#         result = max(result, 0)
#         result = min(result, 720)

#         # Calculate and log MSE after prediction using training data
#         X_train = np.array([
#             [d['GrainTemperature'], d['GrainMoisture'], d['RoomTemperature'], d['Weight']]
#             for d in self.training_data
#         ])
#         y_actual = np.array([d['DryingTime'] for d in self.training_data])
#         mse = self.calculate_mse(X_train, y_actual)
#         logger.info(f"Prediction successful: {result:.2f} minutes, MSE on training data: {mse:.4f}")

#         return result if not np.isnan(result) else 0.0
