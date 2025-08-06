import logging
import requests
import sys
from flask import Flask, request, jsonify
from model import PredictionModel
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class PredictionService:
    def __init__(self, model, laravel_api_url="http://127.0.0.1:3333/api/prediction/receive"):
        self.model = model
        self.laravel_api_url = laravel_api_url
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))

    def process_sensor_data(self, data):
        try:
            required_keys = ['process_id', 'grain_type_id', 'suhu_gabah', 'kadar_air_gabah', 'suhu_ruangan', 'suhu_pembakaran', 'status_pengaduk', 'kadar_air_target', 'weight', 'timestamp']
            if not all(key in data for key in required_keys):
                logger.error("Invalid sensor data format")
                return {"error": "Invalid sensor data format"}, 400

            # Validasi nilai non-null
            if any(data[key] is None for key in ['suhu_gabah', 'kadar_air_gabah', 'suhu_ruangan', 'suhu_pembakaran', 'status_pengaduk', 'kadar_air_target']):
                logger.error("Missing required sensor data")
                return {"error": "Missing required sensor data"}, 400

            process_id = data['process_id']
            grain_type_id = data['grain_type_id']
            weight = data['weight']

            if weight < 100:
                logger.error(f"Invalid grain weight: {weight} kg")
                return {"error": "Invalid grain weight"}, 400

            prediction_input = {
                'GrainTypeId': float(grain_type_id),
                'GrainTemperature': float(data['suhu_gabah']),
                'GrainMoisture': float(data['kadar_air_gabah']),
                'RoomTemperature': float(data['suhu_ruangan']),
                'BurningTemperature': float(data['suhu_pembakaran']),
                'StirrerStatus': 1 if data['status_pengaduk'] else 0,
                'TargetMoisture': float(data['kadar_air_target']), 
                'Weight': float(weight)
            }

            y_pred = self.model.predict(prediction_input)
            y_pred = float(y_pred)  # Konversi ke skalar float
            logger.info(f"Prediction: {y_pred:.2f} minutes (ID: {process_id})")

            laravel_payload = {
                'process_id': process_id,
                'grain_type_id': grain_type_id,
                'suhu_gabah': float(data['suhu_gabah']),
                'kadar_air_gabah': float(data['kadar_air_gabah']),
                'suhu_ruangan': float(data['suhu_ruangan']),
                'suhu_pembakaran': float(data['suhu_pembakaran']),
                'status_pengaduk': bool(data['status_pengaduk']),
                'predicted_drying_time': y_pred,
                'weight': float(weight),
                'timestamp': data['timestamp']
            }

            try:
                response = self.session.post(self.laravel_api_url, json=laravel_payload, timeout=15)
                response.raise_for_status()
                logger.info(f"Data sent to Laravel (ID: {process_id}, Estimated Duration: {y_pred:.2f} minutes)")
                return {"message": "Prediction processed and sent to Laravel", "predicted_drying_time": y_pred}, 200
            except requests.RequestException as e:
                logger.error(f"Failed to send data to Laravel: {e} (ID: {process_id})")
                return {"error": "Failed to send data to Laravel"}, 500

        except Exception as e:
            logger.error(f"Error processing sensor data: {e} (ID: {process_id})")
            return {"error": "Error processing sensor data"}, 500

@app.route('/predict-now', methods=['POST'])
def process_sensor_data():
    data = request.get_json()
    result, status_code = app.prediction_service.process_sensor_data(data)
    return jsonify(result), status_code

def run_flask(prediction_service):
    app.prediction_service = prediction_service
    cli = sys.modules['flask.cli']
    cli.show_server_banner = lambda *x: None
    app.run(host='0.0.0.0', port=5000, debug=False)