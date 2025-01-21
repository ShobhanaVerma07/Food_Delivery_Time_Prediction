import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging
from pathlib import Path

TARGET = "time_taken"

# Create logger
logger = logging.getLogger("data_preparation")
logger.setLevel(logging.INFO)

# Console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Add handler to logger
logger.addHandler(handler)

# Create a formatter
formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# Add formatter to handler
handler.setFormatter(formatter)


def load_data(data_path: Path) -> pd.DataFrame:
    """Load the dataset from the specified path."""
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully from {data_path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def split_data(data: pd.DataFrame, test_size: float, random_state: int):
    """Split the data into train and test sets."""
    try:
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=random_state
        )
        logger.info(f"Data split successfully: {len(train_data)} train rows, {len(test_data)} test rows")
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise


def read_params(file_path: Path):
    """Read parameters from the YAML configuration file."""
    try:
        with open(file_path, "r") as f:
            params_file = yaml.safe_load(f)
        logger.info(f"Parameters loaded successfully from {file_path}")
        return params_file
    except FileNotFoundError:
        logger.error(f"Parameter file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading parameters: {e}")
        raise


def save_data(data: pd.DataFrame, save_path: Path) -> None:
    """Save the DataFrame to a specified path."""
    try:
        data.to_csv(save_path, index=False)
        logger.info(f"Data saved successfully at {save_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise


if __name__ == "__main__":
    try:
        # Set file paths
        root_path = Path(__file__).resolve().parent.parent.parent
        data_path = root_path / "data" / "cleaned" / "swiggy_cleaned.csv"
        save_data_dir = root_path / "data" / "interim"
        save_data_dir.mkdir(exist_ok=True, parents=True)

        train_filename = "train.csv"
        test_filename = "test.csv"
        save_train_path = save_data_dir / train_filename
        save_test_path = save_data_dir / test_filename
        params_file_path = root_path / "params.yaml"

        # Load the cleaned data
        df = load_data(data_path)

        # Read the parameters
        parameters = read_params(params_file_path)["Data_Preparation"]
        test_size = parameters["test_size"]
        random_state = parameters["random_state"]

        # Split into train and test data
        train_data, test_data = split_data(df, test_size=test_size, random_state=random_state)

        # Save the train and test data
        save_data(train_data, save_train_path)
        save_data(test_data, save_test_path)

        logger.info("Data preparation process completed successfully.")
    except Exception as main_e:
        logger.error(f"Data preparation process failed: {main_e}")
        raise
