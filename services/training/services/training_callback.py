"""
Training Callback for Real-time Logging
Custom callback to log training progress in real-time
"""

import logging
from typing import Dict, Any, Optional
from transformers import TrainerCallback, TrainerState, TrainerControl
from services.training_logs_service import TrainingLogsService

logger = logging.getLogger(__name__)

class TrainingProgressCallback(TrainerCallback):
    """Custom callback to log training progress in real-time"""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.current_epoch = 0
        self.total_epochs = 0
        self.step_count = 0
        
    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called when training begins"""
        self.total_epochs = args.num_train_epochs
        logger.info(f"Training started for job {self.job_id} - {self.total_epochs} epochs")
        
    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of each epoch"""
        self.current_epoch = int(state.epoch) + 1
        logger.info(f"Starting epoch {self.current_epoch}/{self.total_epochs} for job {self.job_id}")
        
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs: Optional[Dict[str, float]] = None, **kwargs):
        """Called when logs are generated during training"""
        if logs is None:
            return
            
        # Extract key metrics
        epoch = logs.get('epoch', self.current_epoch)
        step = logs.get('step', state.global_step)
        loss = logs.get('train_loss', logs.get('loss'))
        eval_loss = logs.get('eval_loss')
        eval_accuracy = logs.get('eval_accuracy')
        eval_f1 = logs.get('eval_f1')
        learning_rate = logs.get('learning_rate')
        
        # Create log message
        log_parts = []
        if epoch is not None:
            log_parts.append(f"Epoch {epoch:.1f}")
        if step is not None:
            log_parts.append(f"Step {step}")
        if loss is not None:
            log_parts.append(f"Loss: {loss:.4f}")
        if eval_loss is not None:
            log_parts.append(f"Eval Loss: {eval_loss:.4f}")
        if eval_accuracy is not None:
            log_parts.append(f"Eval Accuracy: {eval_accuracy:.4f}")
        if eval_f1 is not None:
            log_parts.append(f"Eval F1: {eval_f1:.4f}")
        if learning_rate is not None:
            log_parts.append(f"LR: {learning_rate:.2e}")
        
        message = " | ".join(log_parts)
        
        # Log to database
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, create a task
                asyncio.create_task(
                    TrainingLogsService.log_training_event(
                        self.job_id, "INFO", "model_trainer", 
                        message, 
                        {"metrics": logs, "epoch": epoch, "step": step}
                    )
                )
            else:
                # If not in async context, run directly
                loop.run_until_complete(
                    TrainingLogsService.log_training_event(
                        self.job_id, "INFO", "model_trainer", 
                        message, 
                        {"metrics": logs, "epoch": epoch, "step": step}
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to log training progress: {e}")
        
        # Also log to console
        logger.info(f"[{self.job_id}] {message}")
        
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, logs: Optional[Dict[str, float]] = None, **kwargs):
        """Called after evaluation"""
        if logs is None:
            return
            
        epoch = logs.get('epoch', self.current_epoch)
        eval_loss = logs.get('eval_loss')
        eval_accuracy = logs.get('eval_accuracy')
        eval_f1 = logs.get('eval_f1')
        
        message = f"Evaluation completed - Epoch {epoch:.1f}"
        if eval_loss is not None:
            message += f" | Eval Loss: {eval_loss:.4f}"
        if eval_accuracy is not None:
            message += f" | Eval Accuracy: {eval_accuracy:.4f}"
        if eval_f1 is not None:
            message += f" | Eval F1: {eval_f1:.4f}"
        
        # Log to database
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(
                    TrainingLogsService.log_training_event(
                        self.job_id, "INFO", "model_trainer", 
                        message, 
                        {"evaluation_metrics": logs, "epoch": epoch}
                    )
                )
            else:
                loop.run_until_complete(
                    TrainingLogsService.log_training_event(
                        self.job_id, "INFO", "model_trainer", 
                        message, 
                        {"evaluation_metrics": logs, "epoch": epoch}
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to log evaluation results: {e}")
        
        logger.info(f"[{self.job_id}] {message}")
        
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called when training ends"""
        logger.info(f"Training completed for job {self.job_id} after {state.global_step} steps")
