import os
from pathlib import Path
from src.TextSummarizer.constants import *
from src.TextSummarizer.utils.common import read_yaml, create_directories
from src.TextSummarizer.entity import (DataIngestionConfig,
                                       DataValidationConfig,
                                       DataTransformationConfig,
                                       ModelTrainerConfig,
                                       ModelEvaluationConfig)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent  # Should point to the root of the project
CONFIG_FILE_PATH = ROOT_DIR / "configuration/config.yaml"  # This should point to config.yaml in the 'configuration' folder
PARAMS_FILE_PATH = ROOT_DIR / "params.yaml"  # Assuming params.yaml is in the root folder

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        # Check if config file exists and log the paths
        if not CONFIG_FILE_PATH.exists():
            raise FileNotFoundError(f"Config file not found at {CONFIG_FILE_PATH}")

        # Check if params.yaml exists
        if not PARAMS_FILE_PATH.exists():
            raise FileNotFoundError(f"Params file not found at {PARAMS_FILE_PATH}")

        # Create 'artifacts' directory if it doesn't exist
        artifacts_dir = ROOT_DIR / "artifacts"
        create_directories([artifacts_dir])

        # Load configuration and parameters
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        # Ensure directories specified in the config are created
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        # Ensure directories for data ingestion are created
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        # Ensure directories for data validation are created
        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        # Ensure directories for data transformation are created
        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_name=config.tokenizer_name
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments

        # Ensure directories for model trainer are created
        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_ckpt=config.model_ckpt,
            num_train_epochs=params.num_train_epochs,
            warmup_steps=params.warmup_steps,
            per_device_train_batch_size=params.per_device_train_batch_size,
            weight_decay=params.weight_decay,
            logging_steps=params.logging_steps,
            evaluation_strategy=params.evaluation_strategy,
            eval_steps=params.evaluation_strategy,
            save_steps=params.save_steps,
            gradient_accumulation_steps=params.gradient_accumulation_steps
        )

        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        # Ensure directories for model evaluation are created
        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_path=config.model_path,
            tokenizer_path=config.tokenizer_path,
            metric_file_name=config.metric_file_name
        )

        return model_evaluation_config
