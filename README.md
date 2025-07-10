# Bed Dryer ML Prediction

This project implements a Linear Regression model in Python to predict the drying time of rice grains in a bed dryer based on grain temperature, moisture content, and room temperature. The model integrates with MQTT for real-time data processing.

## Prerequisites
- Python 3.12.3
- Virtual environment (recommended)

## Setup
1. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure training-data.json** is in the `data/` folder.

4. **Configure MQTT**:
   - Update `broker` and `port` in `src/main.py` if your MQTT broker is not `localhost:1883`.
   - Ensure topics (`bed_dryer/sensors`, `bed_dryer/prediction`) match your setup.

## Running the Application
```bash
python src/main.py
```

## Project Structure
- `data/training-data.json`: Training data (60 samples).
- `src/model.py`: Linear Regression model training and prediction.
- `src/mqtt_service.py`: MQTT communication.
- `src/main.py`: Main application.
- `requirements.txt`: Dependencies.

## Notes
- The model uses TensorFlow for Linear Regression, trained with 300 epochs and Adam optimizer.
- Predictions are capped at 0–480 minutes.
- If grain moisture is ≤ 14%, drying time is 0 minutes.
- Logs are generated for debugging.

## Example Input
```json
{
  "GrainTemperature": 45.2,
  "GrainMoisture": 25.0,
  "RoomTemperature": 30.8
}
```

Expected output: ~262 minutes (~4 hours 22 minutes).