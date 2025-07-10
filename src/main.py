import logging
from model import PredictionModel
from prediction_service import PredictionService, run_flask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        model = PredictionModel()
        logger.info("Prediction model initialized")

        prediction_service = PredictionService(
            model=model,
            laravel_api_url="http://127.0.0.1:3333/api/prediction/receive"
        )
        logger.info("Prediction service initialized")

        logger.info("Starting Flask server on http://127.0.0.1:5000")
        run_flask(prediction_service)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()