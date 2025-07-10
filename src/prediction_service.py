import logging
import requests
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
            required_keys = ['process_id', 'grain_type_id', 'points', 'weight', 'timestamp']
            if not all(key in data for key in required_keys) or not isinstance(data['points'], list):
                logger.error("Invalid sensor data format")
                return {"error": "Invalid sensor data format"}, 400

            points = data['points']
            grain_temps = [p['grain_temperature'] for p in points if p['grain_temperature'] is not None]
            grain_moists = [p['grain_moisture'] for p in points if p['grain_moisture'] is not None]
            room_temps = [p['room_temperature'] for p in points if p['room_temperature'] is not None]
            burning_temps = [p['burning_temperature'] for p in points if p['burning_temperature'] is not None]
            stirrer_statuses = [1 if p['stirrer_status'] else 0 for p in points if p['stirrer_status'] is not None]

            if not (grain_temps and grain_moists and room_temps and burning_temps and stirrer_statuses):
                logger.error("Missing required sensor data")
                return {"error": "Missing required sensor data"}, 400

            avg_temp = round(sum(grain_temps) / len(grain_temps), 1)
            avg_moist = round(sum(grain_moists) / len(grain_moists), 1)
            avg_room_temp = round(sum(room_temps) / len(room_temps), 1)
            burning_temp = burning_temps[0]  # Ambil nilai tunggal
            stirrer_status = stirrer_statuses[0]  # Ambil nilai tunggal
            grain_type_id = data['grain_type_id']
            weight = data['weight']
            process_id = data['process_id']

            if weight < 100:
                logger.error(f"Invalid grain weight: {weight} kg")
                return {"error": "Invalid grain weight"}, 400

            prediction_input = {
                'GrainTypeId': float(grain_type_id),
                'GrainTemperature': avg_temp,
                'GrainMoisture': avg_moist,
                'RoomTemperature': avg_room_temp,
                'Weight': weight,
                'BurningTemperature': burning_temp,
                'StirrerStatus': stirrer_status
            }

            predicted_time = self.model.predict(prediction_input)  # Mulai Predict
            logger.info(f"Prediction: {predicted_time:.2f} minutes (ID: {process_id})")

            laravel_payload = {
                'process_id': process_id,
                'grain_type_id': grain_type_id,
                'points': points,
                'avg_grain_temperature': avg_temp,
                'avg_grain_moisture': avg_moist,
                'burning_temperature': burning_temp,
                'stirrer_status': bool(stirrer_status),
                'predicted_drying_time': predicted_time,
                'weight': weight,
                'timestamp': data['timestamp']
            }

            try:
                response = self.session.post(self.laravel_api_url, json=laravel_payload, timeout=15)  # Kirim data ke Laravel
                response.raise_for_status()
                logger.info(f"Data sent to Laravel (ID: {process_id}, Estimated Duration: {predicted_time:.2f} minutes)")
                return {"message": "Prediction processed and sent to Laravel", "predicted_drying_time": predicted_time}, 200
            except requests.RequestException as e:
                logger.error(f"Failed to send data to Laravel: {e} (ID: {process_id})")
                return {"error": "Failed to send data to Laravel"}, 500

        except Exception as e:
            logger.error(f"Error processing sensor data: {e} (ID: {process_id})")
            return {"error": "Error processing sensor data"}, 500

@app.route('/process-sensor-data', methods=['POST'])
def process_sensor_data():
    data = request.get_json()
    result, status_code = app.prediction_service.process_sensor_data(data)
    return jsonify(result), status_code

def run_flask(prediction_service):
    app.prediction_service = prediction_service
    app.run(host='0.0.0.0', port=5000, debug=False)