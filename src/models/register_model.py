import mlflow
import dagshub
import json
from pathlib import Path
from mlflow import MlflowClient
import logging

# Create logger
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

# Console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Add handler to logger
logger.addHandler(handler)

# Initialize Dagshub
dagshub.init(repo_owner='ShobhanaVerma07', 
             repo_name='Food_Delivery_Time_Prediction', 
             mlflow=True)

# Set the MLflow tracking server
mlflow.set_tracking_uri("https://dagshub.com/ShobhanaVerma07/Food_Delivery_Time_Prediction.mlflow")

def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info

if __name__ == "__main__":
    # Load model information
    root_path = Path(__file__).parent.parent.parent
    run_info_path = root_path / "run_information.json"
    run_info = load_model_information(run_info_path)

    # Register the model
    run_id = run_info["run_id"]
    model_name = run_info["model_name"]
    model_registry_path = f"runs:/{run_id}/{model_name}"

    # Register the model
    model_version = mlflow.register_model(model_uri=model_registry_path, name=model_name)

    # Get the model version
    registered_model_version = model_version.version
    registered_model_name = model_version.name
    logger.info(f"The latest model version in model registry is {registered_model_version}")

    # Transition the model to Production stage
    client = MlflowClient()
    client.transition_model_version_stage(
        name=registered_model_name,
        version=registered_model_version,
        stage="Production"
    )
    logger.info("Model pushed to Production stage")