import pandas as pd
import joblib
import logging
import mlflow
import dagshub
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import json

# Initialize dagshub
dagshub.init(repo_owner='ShobhanaVerma07', 
            repo_name='Food_Delivery_Time_Prediction', 
            mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/ShobhanaVerma07/Food_Delivery_Time_Prediction.mlflow")
mlflow.set_experiment("DVC Pipeline")

TARGET = "time_taken"

# Logger setup
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_data(data_path: Path) -> pd.DataFrame:
   try:
       df = pd.read_csv(data_path)
       logger.info(f"Data loaded from {data_path}")
       return df
   except FileNotFoundError:
       logger.error("File not found")
       raise

def make_X_and_y(data: pd.DataFrame, target_column: str):
   X = data.drop(columns=[target_column])
   y = data[target_column]
   return X, y

def load_model(model_path: Path):
   return joblib.load(model_path)

def save_model_info(save_json_path, run_id, artifact_path, model_name):
   info_dict = {
       "run_id": run_id,
       "artifact_path": artifact_path,
       "model_name": model_name
   }
   with open(save_json_path, "w") as f:
       json.dump(info_dict, f, indent=4)

def save_local_models(model, preprocessor, save_dir: Path):
   save_dir.mkdir(exist_ok=True)
   joblib.dump(model, save_dir / "model.pkl")
   joblib.dump(preprocessor, save_dir / "preprocessor.pkl")
   logger.info(f"Models saved locally in {save_dir}")

if __name__ == "__main__":
   root_path = Path(__file__).parent.parent.parent
   train_data_path = root_path / "data" / "processed" / "train_trans.csv"
   test_data_path = root_path / "data" / "processed" / "test_trans.csv"
   model_path = root_path / "models" / "model.joblib"
   preprocessor_path = root_path / "models" / "preprocessor.joblib"
   
   # Load data and model
   train_data = load_data(train_data_path)
   test_data = load_data(test_data_path)
   X_train, y_train = make_X_and_y(train_data, TARGET)
   X_test, y_test = make_X_and_y(test_data, TARGET)
   model = load_model(model_path)
   preprocessor = load_model(preprocessor_path)
   
   # Make predictions
   y_train_pred = model.predict(X_train)
   y_test_pred = model.predict(X_test)
   
   # Calculate metrics
   train_mae = mean_absolute_error(y_train, y_train_pred)
   test_mae = mean_absolute_error(y_test, y_test_pred)
   train_r2 = r2_score(y_train, y_train_pred)
   test_r2 = r2_score(y_test, y_test_pred)
   
   cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                             scoring="neg_mean_absolute_error", n_jobs=-1)
   mean_cv_score = -cv_scores.mean()
   
   # MLflow logging
   with mlflow.start_run() as run:
       mlflow.set_tag("model", "Food Delivery Time Regressor")
       mlflow.log_params(model.get_params())
       mlflow.log_metric("train_mae", train_mae)
       mlflow.log_metric("test_mae", test_mae)
       mlflow.log_metric("train_r2", train_r2)
       mlflow.log_metric("test_r2", test_r2)
       mlflow.log_metric("mean_cv_score", mean_cv_score)
       
       for num, score in enumerate(-cv_scores):
           mlflow.log_metric(f"CV {num}", score)
           
       train_data_input = mlflow.data.from_pandas(train_data, targets=TARGET)
       test_data_input = mlflow.data.from_pandas(test_data, targets=TARGET)
       mlflow.log_input(dataset=train_data_input, context="training")
       mlflow.log_input(dataset=test_data_input, context="validation")
       
       model_signature = mlflow.models.infer_signature(
           model_input=X_train.sample(20, random_state=42),
           model_output=model.predict(X_train.sample(20, random_state=42)))
           
       mlflow.sklearn.log_model(model, "delivery_time_pred_model", signature=model_signature)
       
       for artifact in ["stacking_regressor.joblib", "power_transformer.joblib", "preprocessor.joblib"]:
           mlflow.log_artifact(root_path / "models" / artifact)
           
       artifact_uri = mlflow.get_artifact_uri()
       
   # Save local copies
   save_dir = root_path / "saved_models"
   save_local_models(model, preprocessor, save_dir)
   
   # Save run info
   run_id = run.info.run_id
   save_json_path = root_path / "run_information.json"
   save_model_info(save_json_path, run_id, artifact_uri, "delivery_time_pred_model")