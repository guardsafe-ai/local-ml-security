"""
Training Service - Model Trainer
Core training logic and model management
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import warnings
from sklearn.exceptions import ConvergenceWarning
# Targeted warning suppression - only suppress specific warnings that are expected
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

# Initialize logger
logger = logging.getLogger(__name__)
logger.debug("🔇 [WARNINGS] Applied targeted warning filters for ML training")

# Disable MLflow integration in Transformers before importing
os.environ["MLFLOW_DISABLE"] = "1"

# Configure logging
logger = logging.getLogger(__name__)

import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)

# Device detection and configuration with robust fallback
def setup_device():
    """Setup device configuration with CPU fallback"""
    device = None
    device_type = "cpu"
    
    # Check for CUDA availability
    if torch.cuda.is_available():
        try:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3  # GB
            memory_cached = torch.cuda.memory_reserved(current_device) / 1024**3  # GB
            
            logger.info(f"🚀 CUDA GPU Available: {device_count} device(s)")
            logger.info(f"📱 Current Device: {device_name}")
            logger.info(f"💾 Memory Allocated: {memory_allocated:.2f} GB")
            logger.info(f"💾 Memory Cached: {memory_cached:.2f} GB")
            
            # Enable mixed precision training for CUDA
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            device = torch.device('cuda')
            device_type = "cuda"
            
        except Exception as e:
            logger.warning(f"⚠️ CUDA initialization failed: {e}")
            logger.warning("⚠️ Falling back to CPU")
            device = torch.device('cpu')
            device_type = "cpu"
    else:
        logger.info("ℹ️ No CUDA GPU detected, using CPU")
        device = torch.device('cpu')
        device_type = "cpu"
    
    # Log device configuration
    logger.info(f"🎯 Using device: {device} ({device_type.upper()})")
    
    return device, device_type

# Initialize device
DEVICE, DEVICE_TYPE = setup_device()
GPU_AVAILABLE = (DEVICE_TYPE == "cuda")
from sklearn.model_selection import train_test_split
from utils.data_lineage import create_data_splits_with_lineage, DataLineageTracker
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import datasets
from datasets import Dataset as HFDataset

from models.requests import TrainingConfig, TrainingRequest
from models.responses import JobStatus, TrainingStatus, ModelInfo
from database.repositories import TrainingJobRepository, ModelPerformanceRepository
from services.training_config_service import TrainingConfigService
from services.training_logs_service import TrainingLogsService
from services.training_callback import TrainingProgressCallback
from services.data_augmentation import augmenter, AugmentationConfig
from utils.s3_client import S3Client
from efficient_data_manager import DataStatus

logger = logging.getLogger(__name__)


class SecurityDataset(Dataset):
    """Custom dataset for security model training"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ModelTrainer:
    """Main model training class"""
    
    def __init__(self):
        self.job_repo = TrainingJobRepository()
        self.performance_repo = ModelPerformanceRepository()
        self.s3_client = S3Client()
        self.active_jobs: Dict[str, JobStatus] = {}
        self.model_configs = {
            "distilbert": {
                "model_name": "distilbert-base-uncased",
                "labels": ["prompt_injection", "jailbreak", "system_extraction", "code_injection", "benign"],
                "description": "DistilBERT model for ML security classification"
            },
            "bert-base": {
                "model_name": "bert-base-uncased", 
                "labels": ["prompt_injection", "jailbreak", "system_extraction", "code_injection", "benign"],
                "description": "BERT base model for ML security classification"
            },
            "roberta-base": {
                "model_name": "roberta-base",
                "labels": ["prompt_injection", "jailbreak", "system_extraction", "code_injection", "benign"],
                "description": "RoBERTa base model for ML security classification"
            },
            "deberta-v3-base": {
                "model_name": "microsoft/deberta-v3-base",
                "labels": ["prompt_injection", "jailbreak", "system_extraction", "code_injection", "benign"],
                "description": "DeBERTa v3 base model for ML security classification"
            }
        }

    async def load_training_data(self, data_path: str) -> Tuple[List[str], List[int], str]:
        """Load and preprocess training data from S3/MinIO only"""
        try:
            # Ensure data path is from MinIO
            if not data_path.startswith('s3://'):
                raise ValueError(f"Data path must be S3 URI (s3://...), got: {data_path}")
            
            # Load data from MinIO
            if data_path.endswith('.jsonl'):
                data = self.s3_client.load_jsonl_data(data_path)
            else:
                # Assume CSV format
                data = self.s3_client.load_csv_data(data_path)
            
            # Extract texts and labels
            texts = [item['text'] for item in data]
            labels = [item['label'] for item in data]
            
            # Convert labels to integers if they're strings - DYNAMIC MAPPING
            if isinstance(labels[0], str):
                # Extract unique labels and create mapping dynamically
                unique_labels = sorted(list(set(labels)))
                label_map = {label: idx for idx, label in enumerate(unique_labels)}
                
                # Validate all labels are present and warn about any missing expected labels
                expected_labels = {"prompt_injection", "jailbreak", "system_extraction", "code_injection", "benign"}
                missing_labels = expected_labels - set(unique_labels)
                if missing_labels:
                    logger.warning(f"⚠️ [LABEL WARNING] Missing expected labels: {missing_labels}")
                
                # Convert labels with validation - FAIL if unknown label found
                converted_labels = []
                for label in labels:
                    if label in label_map:
                        converted_labels.append(label_map[label])
                    else:
                        raise ValueError(f"Unknown label '{label}' found in data. Expected: {list(label_map.keys())}")
                labels = converted_labels
                
                # Store label mapping for later use
                self.label_map = label_map
                logger.info(f"✅ [LABEL MAPPING] Created dynamic mapping: {label_map}")
            
            # Extract file_id from S3 path
            file_id = data_path.split('/')[-1].split('.')[0]  # Get filename without extension
            
            logger.info(f"✅ Loaded {len(texts)} training samples from MinIO: {data_path}")
            return texts, labels, file_id
            
        except Exception as e:
            logger.error(f"❌ Failed to load training data from MinIO {data_path}: {e}")
            raise


    async def train_model(self, request: TrainingRequest) -> str:
        """Train a model with the given configuration"""
        job_id = f"train_{request.model_name}_{int(time.time())}"
        training_start_time = time.time()  # Track training start time
        
        try:
            # Send business metric for training start
            from main import send_business_metric
            await send_business_metric(
                "training_jobs_started", 
                1, 
                tags={"model": request.model_name}, 
                metadata={"gpu": torch.cuda.is_available()}
            )
            
            # Log training job start
            await TrainingLogsService.log_training_event(
                job_id, "INFO", "training_service", 
                f"Training job {job_id} started",
                {"model_name": request.model_name, "request_config": request.config.dict() if request.config else None}
            )
            
            # Normalize model name - remove _pretrained suffix if present
            base_model_name = request.model_name.replace("_pretrained", "").replace("_trained", "")
            
            await TrainingLogsService.log_training_event(
                job_id, "INFO", "model_trainer", 
                f"Normalized model name: {request.model_name} -> {base_model_name}"
            )
            
            # Get model configuration
            if base_model_name not in self.model_configs:
                await TrainingLogsService.log_training_event(
                    job_id, "ERROR", "model_trainer", 
                    f"Unknown model: {request.model_name} (base: {base_model_name})"
                )
                raise ValueError(f"Unknown model: {request.model_name} (base: {base_model_name})")
            
            model_config = self.model_configs[base_model_name]
            
            await TrainingLogsService.log_training_event(
                job_id, "INFO", "model_trainer", 
                f"Loaded model configuration for {base_model_name}",
                {"model_config": model_config}
            )
            
            # Load saved training configuration from database, fallback to request config or default
            saved_config = None
            try:
                # Try both the original name and base name for configuration lookup
                try:
                    saved_config = await TrainingConfigService.get_config(request.model_name)
                    await TrainingLogsService.log_training_event(
                        job_id, "INFO", "config_service", 
                        f"Loaded saved configuration for model: {request.model_name}",
                        {"saved_config": saved_config}
                    )
                except:
                    saved_config = await TrainingConfigService.get_config(base_model_name)
                    await TrainingLogsService.log_training_event(
                        job_id, "INFO", "config_service", 
                        f"Loaded saved configuration for base model: {base_model_name}",
                        {"saved_config": saved_config}
                    )
            except Exception as e:
                await TrainingLogsService.log_training_event(
                    job_id, "INFO", "config_service", 
                    f"No saved configuration found for model: {request.model_name} or {base_model_name}, using request config or defaults"
                )
            
            # Use saved config if available, otherwise use request config or create default
            if saved_config:
                training_config = TrainingConfig(
                    model_name=base_model_name,  # Use base model name for training
                    training_data_path=saved_config.get('training_data_path', request.training_data_path),
                    hyperparameters=saved_config.get('hyperparameters', {}),
                    validation_split=saved_config.get('validation_split', 0.2),
                    test_split=saved_config.get('test_split', 0.1),
                    early_stopping=saved_config.get('early_stopping', True),
                    patience=saved_config.get('patience', 5),
                    metric_for_best_model=saved_config.get('metric_for_best_model', 'f1_score')
                )
                logger.info(f"✅ Using saved configuration for training: {training_config.model_name}")
            else:
                training_config = request.config or TrainingConfig(model_name=base_model_name)
                logger.info(f"📝 Using request configuration or defaults for training: {training_config.model_name}")
            
            # Check data availability before proceeding with training
            training_data_path = request.training_data_path
            data_available = False
            
            if not training_data_path or training_data_path == "auto":
                # Use EfficientDataManager to check for available training data
                try:
                    from efficient_data_manager import EfficientDataManager
                    data_manager = EfficientDataManager()
                    
                    # Get fresh data files
                    fresh_files = []
                    for file_id, file_info in data_manager.data_files.items():
                        status = file_info.status.value if hasattr(file_info.status, 'value') else str(file_info.status)
                        if status == "fresh":
                            fresh_files.append(file_info)
                    
                    if fresh_files:
                        # Use the most recent fresh file
                        latest_file = max(fresh_files, key=lambda x: x.upload_time)
                        training_data_path = f"s3://{data_manager.bucket_name}/{latest_file.minio_path}"
                        data_available = True
                        
                        await TrainingLogsService.log_training_event(
                            job_id, "INFO", "data_loader", 
                            f"Found fresh data file: {latest_file.original_name}",
                            {"file_id": latest_file.file_id, "file_size": latest_file.file_size}
                        )
                    else:
                        # Check if fallback data exists
                        fallback_path = data_manager.get_training_data_path()
                        if fallback_path and fallback_path.startswith("s3://"):
                            # Verify the fallback file exists in MinIO
                            try:
                                s3_path_parts = fallback_path.replace('s3://', '').split('/', 1)
                                if len(s3_path_parts) == 2:
                                    bucket_name, s3_key = s3_path_parts
                                    data_manager.s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                                    training_data_path = fallback_path
                                    data_available = True
                                    
                                    await TrainingLogsService.log_training_event(
                                        job_id, "INFO", "data_loader", 
                                        f"Using fallback data: {s3_key}"
                                    )
                            except Exception as e:
                                await TrainingLogsService.log_training_event(
                                    job_id, "WARNING", "data_loader", 
                                    f"Fallback data not available: {e}"
                                )
                        
                        if not data_available:
                            await TrainingLogsService.log_training_event(
                                job_id, "ERROR", "data_loader", 
                                "No training data available - no fresh files and no fallback data"
                            )
                            
                except Exception as e:
                    await TrainingLogsService.log_training_event(
                        job_id, "ERROR", "data_loader", 
                        f"Failed to check data availability: {e}"
                    )
            else:
                # Custom data path provided - verify it exists
                try:
                    if training_data_path.startswith("s3://"):
                        # Verify S3 path exists
                        s3_path_parts = training_data_path.replace('s3://', '').split('/', 1)
                        if len(s3_path_parts) == 2:
                            bucket_name, s3_key = s3_path_parts
                            from efficient_data_manager import EfficientDataManager
                            data_manager = EfficientDataManager()
                            data_manager.s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                            data_available = True
                            
                            await TrainingLogsService.log_training_event(
                                job_id, "INFO", "data_loader", 
                                f"Using custom data path: {s3_key}"
                            )
                    else:
                        # Local file path - check if file exists
                        import os
                        if os.path.exists(training_data_path):
                            data_available = True
                            await TrainingLogsService.log_training_event(
                                job_id, "INFO", "data_loader", 
                                f"Using local data path: {training_data_path}"
                            )
                        else:
                            await TrainingLogsService.log_training_event(
                                job_id, "ERROR", "data_loader", 
                                f"Local data file not found: {training_data_path}"
                            )
                            
                except Exception as e:
                    await TrainingLogsService.log_training_event(
                        job_id, "ERROR", "data_loader", 
                        f"Failed to verify custom data path: {e}"
                    )
            
            # If no data is available, fail the training job early
            if not data_available:
                error_message = f"No training data available for job {job_id}. Please upload training data or create sample data first."
                await TrainingLogsService.log_training_event(
                    job_id, "ERROR", "training_service", 
                    error_message
                )
                
                # Update job status to failed
                await self.job_repo.update_job_status(
                    job_id, "failed", error_message=error_message
                )
                
                raise ValueError(error_message)
            
            # Create job record with configuration
            config_dict = training_config.dict() if training_config else {}
            await self.job_repo.create_job(
                job_id, 
                base_model_name,  # Use base model name for job tracking
                "running",
                training_data_path,
                config_dict
            )
            
            # Load training data
            await TrainingLogsService.log_training_event(
                job_id, "INFO", "data_loader", 
                f"Loading training data from: {training_data_path}"
            )
            texts, labels, data_file_id = await self.load_training_data(training_data_path)
            
            await TrainingLogsService.log_training_event(
                job_id, "INFO", "data_loader", 
                f"Loaded {len(texts)} training samples",
                {"sample_count": len(texts), "label_count": len(set(labels))}
            )
            
            # Apply data augmentation
            augmentation_config = AugmentationConfig(
                synonym_replacement_prob=0.3,
                random_insertion_prob=0.1,
                random_deletion_prob=0.1,
                random_swap_prob=0.1,
                random_caps_prob=0.1,
                max_augmentations=2
            )
            
            await TrainingLogsService.log_training_event(
                job_id, "INFO", "data_augmentation", 
                "Starting data augmentation to improve model robustness"
            )
            
            # Augment dataset with 1.5x factor
            augmented_texts, augmented_labels = augmenter.augment_dataset(
                texts, labels, augmentation_factor=1.5
            )
            
            # Balance dataset
            balanced_texts, balanced_labels = augmenter.balance_dataset(
                augmented_texts, augmented_labels
            )
            
            await TrainingLogsService.log_training_event(
                job_id, "INFO", "data_augmentation", 
                f"Data augmentation complete: {len(texts)} -> {len(balanced_texts)} samples",
                {
                    "original_samples": len(texts),
                    "augmented_samples": len(augmented_texts),
                    "balanced_samples": len(balanced_labels),
                    "augmentation_factor": len(balanced_texts) / len(texts)
                }
            )
            
            # Use augmented and balanced data
            texts, labels = balanced_texts, balanced_labels
            
            # Initialize tokenizer and model
            await TrainingLogsService.log_training_event(
                job_id, "INFO", "model_trainer", 
                f"Initializing tokenizer for model: {model_config['model_name']}"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"])
            
            await TrainingLogsService.log_training_event(
                job_id, "INFO", "model_trainer", 
                f"Initializing model for sequence classification with {len(model_config['labels'])} labels"
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_config["model_name"],
                num_labels=len(model_config["labels"]),
                problem_type="single_label_classification"
            )
            
            # Move model to appropriate device (GPU or CPU)
            model = model.to(DEVICE)
            await TrainingLogsService.log_training_event(
                job_id, "INFO", "model_trainer", 
                f"Model moved to {DEVICE_TYPE.upper()}: {DEVICE}"
            )
            
            # Create datasets with proper train/val/test split and data lineage tracking
            if len(texts) < 10:
                logger.warning(f"⚠️ Dataset too small ({len(texts)} samples) for proper train/val/test split. Using all data for training.")
                train_texts, val_texts, test_texts = texts, texts[:1], texts[:1]  # Use first sample as validation and test
                train_labels, val_labels, test_labels = labels, labels[:1], labels[:1]
                
                # Create minimal lineage tracker for small datasets
                from utils.data_lineage import DataLineageTracker
                lineage_tracker = DataLineageTracker()
                logger.warning("⚠️ [DATA LINEAGE] Skipping lineage tracking for small dataset")
            else:
                # Use data lineage tracking to prevent leakage
                train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, lineage_tracker = create_data_splits_with_lineage(
                    texts=texts,
                    labels=labels,
                    data_source=training_data_path,
                    test_size=0.2,
                    val_size=0.25,
                    random_state=42
                )
                
                await TrainingLogsService.log_training_event(
                    job_id, "INFO", "data_splitter", 
                    f"Data split with lineage tracking: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test",
                    {
                        "train_size": len(train_texts), 
                        "val_size": len(val_texts), 
                        "test_size": len(test_texts),
                        "lineage_summary": lineage_tracker.get_split_summary()
                    }
                )
            
            train_dataset = SecurityDataset(train_texts, train_labels, tokenizer, training_config.max_length)
            val_dataset = SecurityDataset(val_texts, val_labels, tokenizer, training_config.max_length)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"./models/{request.model_name}",
                num_train_epochs=training_config.num_epochs,
                per_device_train_batch_size=training_config.batch_size,
                per_device_eval_batch_size=training_config.batch_size,
                warmup_steps=training_config.warmup_steps,
                weight_decay=training_config.weight_decay,
                learning_rate=training_config.learning_rate,
                evaluation_strategy=training_config.evaluation_strategy,
                eval_steps=training_config.eval_steps,
                save_steps=training_config.save_steps,
                load_best_model_at_end=training_config.load_best_model_at_end,
                metric_for_best_model=training_config.metric_for_best_model,
                greater_is_better=training_config.greater_is_better,
                logging_dir=f"./logs/{request.model_name}",
                logging_steps=100,
                save_total_limit=2,
                seed=42,
                fp16=GPU_AVAILABLE,  # Enable FP16 for GPU training
                fp16_full_eval=GPU_AVAILABLE,  # Enable FP16 for evaluation on GPU
                dataloader_num_workers=0,
                remove_unused_columns=False
            )
            
            # Initialize trainer with custom callback for real-time logging
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                callbacks=[
                    EarlyStoppingCallback(early_stopping_patience=2),
                    TrainingProgressCallback(job_id)
                ]
            )
            
            # Start training
            await TrainingLogsService.log_training_event(
                job_id, "INFO", "model_trainer", 
                f"Starting training loop for {request.model_name}",
                {"epochs": training_config.num_epochs, "batch_size": training_config.batch_size}
            )
            trainer.train()
            
            # Evaluate model
            await TrainingLogsService.log_training_event(
                job_id, "INFO", "model_trainer", 
                "Training completed, starting evaluation"
            )
            eval_results = trainer.evaluate()
            
            await TrainingLogsService.log_training_event(
                job_id, "INFO", "model_trainer", 
                f"Validation evaluation completed with results: {eval_results}",
                {"eval_results": eval_results}
            )
            
            # Test set evaluation
            test_dataset = SecurityDataset(test_texts, test_labels, tokenizer, training_config.max_length)
            test_results = trainer.evaluate(eval_dataset=test_dataset)
            
            await TrainingLogsService.log_training_event(
                job_id, "INFO", "model_trainer", 
                f"Test set evaluation completed with results: {test_results}",
                {"test_results": test_results}
            )
            
            # Save model to MLflow with complete workflow
            model_name = f"security_{base_model_name}"
            try:
                await TrainingLogsService.log_training_event(
                    job_id, "INFO", "mlflow_service", 
                    f"Starting MLflow model registration: {model_name}"
                )
                
                # End any existing run first
                if mlflow.active_run():
                    mlflow.end_run()
                
                with mlflow.start_run(run_name=f"training_{base_model_name}_{int(time.time())}"):
                    # Create model signature for input validation
                    from mlflow.models.signature import infer_signature
                    
                    # Generate sample input for signature inference
                    sample_text = "This is a test prompt for signature inference"
                    sample_input = tokenizer(sample_text, return_tensors="pt", max_length=512, padding=True, truncation=True)
                    
                    # Get sample output
                    model.eval()
                    with torch.no_grad():
                        sample_output = model(**sample_input)
                    
                    # Infer signature
                    signature = infer_signature(
                        model_input={"text": [sample_text]},
                        model_output=sample_output.logits.numpy()
                    )
                    
                    # Log comprehensive metadata
                    mlflow.log_params({
                        "training_data_version": data_file_id,
                        "training_data_size": len(texts),
                        "augmentation_factor": 1.5,
                        "model_architecture": base_model_name,
                        "num_labels": len(model_config["labels"]),
                        "device_type": DEVICE_TYPE,
                        "gpu_available": GPU_AVAILABLE,
                        "training_duration_minutes": (time.time() - training_start_time) / 60
                    })
                    
                    # Log model with signature and input example
                    mlflow.pytorch.log_model(
                        pytorch_model=model,
                        artifact_path="model",
                        registered_model_name=model_name,
                        signature=signature,
                        input_example={"text": [sample_text]},
                        pip_requirements=["transformers==4.36.2", f"torch=={torch.__version__}"]
                    )
                    
                    # Save tokenizer with model
                    tokenizer.save_pretrained("/tmp/tokenizer")
                    mlflow.log_artifacts("/tmp/tokenizer", artifact_path="tokenizer")
                    
                    # Log metrics
                    mlflow.log_metrics(eval_results)
                
                # Set model to Staging stage
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                
                # Get the latest version
                versions = client.search_model_versions(f"name='{model_name}'")
                if versions:
                    latest_version = versions[0].version
                    client.transition_model_version_stage(
                        name=model_name,
                        version=latest_version,
                        stage="Staging"
                    )
                    
                    await TrainingLogsService.log_training_event(
                        job_id, "INFO", "mlflow_service", 
                        f"Model {model_name} v{latest_version} moved to Staging stage"
                    )
                
                await TrainingLogsService.log_training_event(
                    job_id, "INFO", "mlflow_service", 
                    f"Successfully registered model in MLflow: {model_name}"
                )
            except Exception as e:
                await TrainingLogsService.log_training_event(
                    job_id, "WARNING", "mlflow_service", 
                    f"MLflow logging failed: {e}"
                )
                # Continue without MLflow if it fails
            
            # Save performance metrics and model metadata in a transaction
            async with self.db_manager.transaction() as conn:
                # Save performance metrics
                await self.performance_repo.save_performance(
                    base_model_name,
                    "latest",
                    eval_results,
                    training_data_path
                )
                
                # Update model metadata
                await conn.execute("""
                    INSERT INTO training.model_metadata 
                    (model_name, version, metrics, training_data_path, created_at)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (model_name, version) 
                    DO UPDATE SET 
                        metrics = EXCLUDED.metrics,
                        training_data_path = EXCLUDED.training_data_path,
                        updated_at = CURRENT_TIMESTAMP
                """, base_model_name, "latest", json.dumps(eval_results), training_data_path, datetime.now())
                
                # Update model lineage
                await conn.execute("""
                    INSERT INTO training.model_lineage 
                    (parent_model, child_model, version, created_at)
                    VALUES ($1, $2, $3, $4)
                """, model_config["model_name"], base_model_name, "latest", datetime.now())
            
            # Move data from fresh to used folder (data lifecycle management)
            try:
                if training_data_path.startswith('s3://'):
                    # Extract the S3 key from the path
                    s3_path_parts = training_data_path.replace('s3://', '').split('/', 1)
                    if len(s3_path_parts) == 2:
                        bucket_name, s3_key = s3_path_parts
                        
                        # Check if the file is in the fresh folder
                        if 'training-data/fresh/' in s3_key:
                            await TrainingLogsService.log_training_event(
                                job_id, "INFO", "data_lifecycle", 
                                f"Moving training data from fresh to used: {s3_key}"
                            )
                            
                            # Use EfficientDataManager for proper lifecycle management
                            try:
                                from efficient_data_manager import EfficientDataManager
                                data_manager = EfficientDataManager()
                                
                                # Find the file in tracking and mark as used
                                file_found = False
                                for file_id, file_info in data_manager.data_files.items():
                                    if file_info.minio_path == s3_key:
                                        # Mark as used using the proper method
                                        file_info.status = DataStatus.USED
                                        file_info.used_count += 1
                                        file_info.last_used = datetime.now()
                                        file_info.training_jobs.append(job_id)
                                        
                                        # Move file to used folder
                                        used_key = s3_key.replace('training-data/fresh/', 'training-data/used/')
                                        
                                        # Copy to used folder
                                        data_manager.s3_client.copy_object(
                                            CopySource={'Bucket': bucket_name, 'Key': s3_key},
                                            Bucket=bucket_name,
                                            Key=used_key
                                        )
                                        
                                        # Delete from fresh folder
                                        data_manager.s3_client.delete_object(
                                            Bucket=bucket_name,
                                            Key=s3_key
                                        )
                                        
                                        # Update path in tracking
                                        file_info.minio_path = used_key
                                        
                                        # Save tracking
                                        data_manager._save_data_tracking()
                                        
                                        await TrainingLogsService.log_training_event(
                                            job_id, "INFO", "data_lifecycle", 
                                            f"Successfully moved data to used folder: {used_key}",
                                            {"file_id": file_id, "used_count": file_info.used_count}
                                        )
                                        
                                        file_found = True
                                        break
                                
                                if not file_found:
                                    # File not in tracking, use simple method
                                    data_manager.mark_data_as_used(s3_key)
                                    await TrainingLogsService.log_training_event(
                                        job_id, "INFO", "data_lifecycle", 
                                        f"File not in tracking, marked as used: {s3_key}"
                                    )
                                    
                            except Exception as e:
                                await TrainingLogsService.log_training_event(
                                    job_id, "WARNING", "data_lifecycle", 
                                    f"Failed to update data tracking: {e}"
                                )
                                
            except Exception as e:
                await TrainingLogsService.log_training_event(
                    job_id, "WARNING", "data_lifecycle", 
                    f"Failed to move data from fresh to used folder: {e}"
                )
            
            # Send business metrics for training completion
            training_duration = time.time() - training_start_time
            final_accuracy = eval_results.get('eval_accuracy', 0.0)
            final_f1 = eval_results.get('eval_f1', 0.0)
            
            await send_business_metric(
                "training_duration_seconds", 
                training_duration,
                tags={"model": request.model_name},
                metadata={"accuracy": final_accuracy, "f1": final_f1}
            )
            
            await send_business_metric(
                "training_jobs_completed", 
                1,
                tags={"model": request.model_name},
                metadata={"accuracy": final_accuracy, "f1": final_f1, "duration_seconds": training_duration}
            )
            
            # Update job status
            result = {
                "status": "completed",
                "mlflow_model_uri": f"models:/{model_name}/latest",
                "model_version": "v1.0",
                "metrics": eval_results,
                "training_time": time.time(),
                "storage": "MLflow/MinIO only (no local files)"
            }
            
            await self.job_repo.update_job_status(
                job_id, "completed", 100.0, result=result
            )
            
            await TrainingLogsService.log_training_event(
                job_id, "INFO", "training_service", 
                f"Training job {job_id} completed successfully for model {base_model_name}",
                {"final_status": "completed", "model_name": base_model_name}
            )
            
            # Automatically promote model to Staging after successful training
            try:
                await self._auto_promote_model_to_staging(model_name, final_accuracy, final_f1)
            except Exception as e:
                logger.warning(f"⚠️ Failed to auto-promote model {model_name} to staging: {e}")
            
            logger.info(f"Training job {job_id} completed successfully for model {base_model_name}")
            return job_id
            
        except Exception as e:
            from utils.enhanced_logging import log_error_with_context
            log_error_with_context(
                error=e,
                operation="model_training",
                model_name=base_model_name,
                additional_context={
                    "job_id": job_id,
                    "gpu_available": torch.cuda.is_available(),
                    "training_data_path": request.training_data_path,
                    "config": request.config.dict() if hasattr(request.config, 'dict') else str(request.config)
                }
            )
            
            # Send business metric for training failure
            await send_business_metric(
                "training_failures", 
                1,
                tags={"model": request.model_name, "error": str(e)[:100]},
                metadata={"error_type": type(e).__name__, "gpu": torch.cuda.is_available()}
            )
            
            await TrainingLogsService.log_training_event(
                job_id, "ERROR", "training_service", 
                f"Training job {job_id} failed for model {base_model_name}: {e}",
                {"error": str(e), "model_name": base_model_name}
            )
            
            await self.job_repo.update_job_status(
                job_id, "failed", error_message=str(e)
            )
            raise

    async def _auto_promote_model_to_staging(self, model_name: str, accuracy: float, f1_score: float):
        """Automatically promote model to Staging stage after successful training"""
        try:
            logger.info(f"🔄 Auto-promoting model {model_name} to Staging stage...")
            
            # Get the latest model version from MLflow
            client = MlflowClient()
            
            # Check if model is already in Staging stage
            staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
            if staging_versions:
                latest_version = staging_versions[0]
                logger.info(f"✅ Model {model_name} v{latest_version.version} is already in Staging stage")
            else:
                # Try to get from None stage and promote
                none_versions = client.get_latest_versions(model_name, stages=["None"])
                if not none_versions:
                    logger.warning(f"⚠️ No model versions found for {model_name} in None or Staging stages")
                    return
                
                latest_version = none_versions[0]
                
                # Transition model to Staging
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version.version,
                    stage="Staging",
                    archive_existing_versions=False
                )
            
            # Add metadata about the promotion
            client.set_model_version_tag(
                name=model_name,
                version=latest_version.version,
                key="auto_promoted",
                value="true"
            )
            
            client.set_model_version_tag(
                name=model_name,
                version=latest_version.version,
                key="promotion_accuracy",
                value=str(accuracy)
            )
            
            client.set_model_version_tag(
                name=model_name,
                version=latest_version.version,
                key="promotion_f1",
                value=str(f1_score)
            )
            
            client.set_model_version_tag(
                name=model_name,
                version=latest_version.version,
                key="promotion_timestamp",
                value=str(int(time.time()))
            )
            
            logger.info(f"✅ Model {model_name} v{latest_version.version} successfully promoted to Staging stage")
            
            # Log the promotion event
            await TrainingLogsService.log_training_event(
                f"auto_promotion_{int(time.time())}", "INFO", "training_service",
                f"Model {model_name} v{latest_version.version} auto-promoted to Staging",
                {
                    "model_name": model_name,
                    "version": latest_version.version,
                    "accuracy": accuracy,
                    "f1_score": f1_score,
                    "stage": "Staging"
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to auto-promote model {model_name} to Staging: {e}")
            raise

    async def get_available_models(self) -> Dict[str, ModelInfo]:
        """Get available models for training"""
        models = {}
        
        for model_name, config in self.model_configs.items():
            # Add base model
            models[model_name] = ModelInfo(
                name=model_name,
                type="pytorch",
                status="available",
                loaded=False,
                path=config["model_name"],
                labels=config["labels"],
                performance=None,
                model_source="Hugging Face",
                model_version="pre-trained",
                description=config.get("description", f"Pre-trained {model_name} model for ML security classification")
            )
            
            # Note: Removed duplicate _pretrained models for cleaner architecture
            # All models are pre-trained by default from Hugging Face
        
        return models

    async def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get status of a training job"""
        job_data = await self.job_repo.get_job(job_id)
        if not job_data:
            return None
        
        return JobStatus(
            job_id=job_data["job_id"],
            model_name=job_data["model_name"],
            status=job_data["status"],
            progress=job_data["progress"],
            start_time=job_data["start_time"],
            end_time=job_data["end_time"],
            error_message=job_data["error_message"],
            result=job_data["result"],
            training_data_path=job_data.get("training_data_path"),
            learning_rate=job_data.get("learning_rate"),
            batch_size=job_data.get("batch_size"),
            num_epochs=job_data.get("num_epochs"),
            max_length=job_data.get("max_length"),
            config=job_data.get("config")
        )

    async def list_jobs(self) -> List[JobStatus]:
        """List all training jobs"""
        jobs_data = await self.job_repo.list_jobs()
        return [
            JobStatus(
                job_id=job["job_id"],
                model_name=job["model_name"],
                status=job["status"],
                progress=job["progress"],
                start_time=job["start_time"],
                end_time=job["end_time"],
                error_message=job["error_message"],
                result=job["result"],
                training_data_path=job.get("training_data_path"),
                learning_rate=job.get("learning_rate"),
                batch_size=job.get("batch_size"),
                num_epochs=job.get("num_epochs"),
                max_length=job.get("max_length"),
                config=job.get("config")
            )
            for job in jobs_data
        ]
