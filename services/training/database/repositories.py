"""
Training Service - Database Repositories
Data access layer for training service
"""

import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from database.async_connection import db_manager

logger = logging.getLogger(__name__)


class TrainingJobRepository:
    """Repository for training job operations"""
    
    async def create_job(self, job_id: str, model_name: str, status: str = "pending", 
                        training_data_path: str = None, config: Dict[str, Any] = None) -> bool:
        """Create a new training job"""
        try:
            # Extract configuration values
            learning_rate = config.get('learning_rate') if config else None
            batch_size = config.get('batch_size') if config else None
            num_epochs = config.get('num_epochs') if config else None
            max_length = config.get('max_length') if config else None
            config_json = json.dumps(config) if config else None
            
            await db_manager.execute_command("""
                INSERT INTO training.training_jobs 
                (job_id, model_name, status, progress, start_time, training_data_path, 
                 learning_rate, batch_size, num_epochs, max_length, config)
                VALUES ($1, $2, $3, 0.0, CURRENT_TIMESTAMP, $4, $5, $6, $7, $8, $9)
            """, job_id, model_name, status, training_data_path, learning_rate, 
                 batch_size, num_epochs, max_length, config_json)
            return True
        except Exception as e:
            logger.error(f"Failed to create training job {job_id}: {e}")
            return False

    async def update_job_status(self, job_id: str, status: str, progress: float = None, 
                              error_message: str = None, result: Dict[str, Any] = None) -> bool:
        """Update training job status and progress"""
        try:
            query = "UPDATE training.training_jobs SET status = $1"
            params = [status]
            param_count = 1
            
            if progress is not None:
                param_count += 1
                query += f", progress = ${param_count}"
                params.append(progress)
            
            if error_message is not None:
                param_count += 1
                query += f", error_message = ${param_count}"
                params.append(error_message)
            
            if result is not None:
                param_count += 1
                query += f", result = ${param_count}"
                params.append(json.dumps(result))
            
            if status in ["completed", "failed"]:
                query += ", end_time = CURRENT_TIMESTAMP"
            
            param_count += 1
            query += f" WHERE job_id = ${param_count}"
            params.append(job_id)
            
            await db_manager.execute_command(query, *params)
            return True
        except Exception as e:
            logger.error(f"Failed to update job {job_id}: {e}")
            return False

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get training job by ID"""
        try:
            result = await db_manager.execute_query("""
                SELECT * FROM training.training_jobs WHERE job_id = $1
            """, job_id)
            return dict(result[0]) if result else None
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            return None

    async def list_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all training jobs"""
        try:
            result = await db_manager.execute_query("""
                SELECT * FROM training.training_jobs 
                ORDER BY created_at DESC 
                LIMIT $1
            """, limit)
            return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            return []

    async def get_jobs_by_model(self, model_name: str) -> List[Dict[str, Any]]:
        """Get training jobs for a specific model"""
        try:
            result = await db_manager.execute_query("""
                SELECT * FROM training.training_jobs 
                WHERE model_name = $1 
                ORDER BY created_at DESC
            """, model_name)
            return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Failed to get jobs for model {model_name}: {e}")
            return []


class ModelPerformanceRepository:
    """Repository for model performance operations"""
    
    async def save_performance(self, model_name: str, version: str, metrics: Dict[str, Any], 
                             test_data_path: str = None) -> bool:
        """Save model performance metrics"""
        try:
            await db_manager.execute_command("""
                INSERT INTO training.model_performance 
                (model_name, version, metrics, test_data_path)
                VALUES ($1, $2, $3, $4)
            """, model_name, version, json.dumps(metrics), test_data_path)
            return True
        except Exception as e:
            logger.error(f"Failed to save performance for {model_name}: {e}")
            return False

    async def get_performance_history(self, model_name: str) -> List[Dict[str, Any]]:
        """Get performance history for a model"""
        try:
            result = await db_manager.execute_query("""
                SELECT * FROM training.model_performance 
                WHERE model_name = $1 
                ORDER BY created_at DESC
            """, model_name)
            return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Failed to get performance history for {model_name}: {e}")
            return []

    async def get_best_performance(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get best performance for a model"""
        try:
            result = await db_manager.execute_query("""
                SELECT * FROM training.model_performance 
                WHERE model_name = $1 
                ORDER BY (metrics->>'eval_f1')::float DESC 
                LIMIT 1
            """, model_name)
            return dict(result[0]) if result else None
        except Exception as e:
            logger.error(f"Failed to get best performance for {model_name}: {e}")
            return None


class RetrainingHistoryRepository:
    """Repository for retraining history operations"""
    
    async def save_retraining(self, model_name: str, trigger_type: str, 
                            performance_before: Dict[str, Any], performance_after: Dict[str, Any],
                            retraining_data_path: str = None) -> bool:
        """Save retraining history"""
        try:
            await db_manager.execute_command("""
                INSERT INTO training.retraining_history 
                (model_name, trigger_type, performance_before, performance_after, retraining_data_path)
                VALUES ($1, $2, $3, $4, $5)
            """, model_name, trigger_type, json.dumps(performance_before), 
                 json.dumps(performance_after), retraining_data_path)
            return True
        except Exception as e:
            logger.error(f"Failed to save retraining history for {model_name}: {e}")
            return False

    async def get_retraining_history(self, model_name: str) -> List[Dict[str, Any]]:
        """Get retraining history for a model"""
        try:
            result = await db_manager.execute_query("""
                SELECT * FROM training.retraining_history 
                WHERE model_name = $1 
                ORDER BY created_at DESC
            """, model_name)
            return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Failed to get retraining history for {model_name}: {e}")
            return []

    async def get_model_lineage(self, model_name: str) -> List[Dict[str, Any]]:
        """Get model lineage (retraining chain)"""
        try:
            result = await db_manager.execute_query("""
                SELECT * FROM training.retraining_history 
                WHERE model_name = $1 
                ORDER BY created_at ASC
            """, model_name)
            return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Failed to get model lineage for {model_name}: {e}")
            return []