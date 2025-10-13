"""
Training Configuration Service
Handles training configuration CRUD operations with database persistence
"""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from database.async_connection import db_manager

logger = logging.getLogger(__name__)

class TrainingConfigService:
    """Service for managing training configurations in the database"""
    
    @staticmethod
    async def get_config(model_name: str) -> Optional[Dict[str, Any]]:
        """Get training configuration for a specific model"""
        try:
            query = """
                SELECT 
                    model_name,
                    training_data_path,
                    hyperparameters,
                    validation_split,
                    test_split,
                    early_stopping,
                    patience,
                    metric_for_best_model,
                    created_at,
                    updated_at
                FROM training_configurations 
                WHERE model_name = $1
            """
            
            result = await db_manager.execute_fetchone(query, model_name)
            
            if result:
                # Convert JSONB hyperparameters to dict if it's a string
                if isinstance(result['hyperparameters'], str):
                    result['hyperparameters'] = json.loads(result['hyperparameters'])
                
                logger.info(f"📋 Retrieved training config for model: {model_name}")
                return result
            else:
                logger.info(f"📋 No training config found for model: {model_name}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get training config for {model_name}: {e}")
            raise
    
    @staticmethod
    async def save_config(model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Save training configuration for a specific model"""
        try:
            # Validate required fields
            required_fields = ['model_name', 'training_data_path', 'hyperparameters']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field: {field}")
            
            # Prepare data for database
            hyperparameters_json = json.dumps(config['hyperparameters'])
            
            # Use UPSERT (INSERT ... ON CONFLICT ... DO UPDATE)
            query = """
                INSERT INTO training_configurations (
                    model_name,
                    training_data_path,
                    hyperparameters,
                    validation_split,
                    test_split,
                    early_stopping,
                    patience,
                    metric_for_best_model,
                    updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP)
                ON CONFLICT (model_name) 
                DO UPDATE SET
                    training_data_path = EXCLUDED.training_data_path,
                    hyperparameters = EXCLUDED.hyperparameters,
                    validation_split = EXCLUDED.validation_split,
                    test_split = EXCLUDED.test_split,
                    early_stopping = EXCLUDED.early_stopping,
                    patience = EXCLUDED.patience,
                    metric_for_best_model = EXCLUDED.metric_for_best_model,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING model_name, updated_at
            """
            
            result = await db_manager.execute_fetchone(
                query,
                model_name,
                config.get('training_data_path', ''),
                hyperparameters_json,
                config.get('validation_split', 0.2),
                config.get('test_split', 0.1),
                config.get('early_stopping', True),
                config.get('patience', 5),
                config.get('metric_for_best_model', 'f1_score')
            )
            
            logger.info(f"💾 Saved training config for model: {model_name}")
            
            return {
                "message": f"Training configuration saved successfully for model: {model_name}",
                "model_name": model_name,
                "updated_at": result['updated_at'].isoformat() if result else datetime.now().isoformat(),
                "config": config
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to save training config for {model_name}: {e}")
            raise
    
    @staticmethod
    async def list_configs() -> Dict[str, Any]:
        """List all saved training configurations"""
        try:
            query = """
                SELECT 
                    model_name,
                    training_data_path,
                    validation_split,
                    test_split,
                    early_stopping,
                    patience,
                    metric_for_best_model,
                    created_at,
                    updated_at
                FROM training_configurations 
                ORDER BY updated_at DESC
            """
            
            results = await db_manager.execute_query(query)
            
            config_list = []
            for row in results:
                config_list.append({
                    "model_name": row['model_name'],
                    "training_data_path": row['training_data_path'],
                    "validation_split": float(row['validation_split']) if row['validation_split'] else 0.2,
                    "test_split": float(row['test_split']) if row['test_split'] else 0.1,
                    "early_stopping": row['early_stopping'],
                    "patience": row['patience'],
                    "metric_for_best_model": row['metric_for_best_model'],
                    "created_at": row['created_at'].isoformat() if row['created_at'] else None,
                    "last_updated": row['updated_at'].isoformat() if row['updated_at'] else None,
                    "has_config": True
                })
            
            logger.info(f"📋 Listed {len(config_list)} training configurations")
            
            return {
                "configurations": config_list,
                "count": len(config_list)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to list training configs: {e}")
            raise
    
    @staticmethod
    async def delete_config(model_name: str) -> Dict[str, Any]:
        """Delete training configuration for a specific model"""
        try:
            query = "DELETE FROM training_configurations WHERE model_name = $1"
            result = await db_manager.execute_command(query, model_name)
            
            if "DELETE 1" in result:
                logger.info(f"🗑️ Deleted training config for model: {model_name}")
                return {"message": f"Training configuration deleted for model: {model_name}"}
            else:
                logger.info(f"📋 No training config found to delete for model: {model_name}")
                return {"message": f"No training configuration found for model: {model_name}"}
                
        except Exception as e:
            logger.error(f"❌ Failed to delete training config for {model_name}: {e}")
            raise
    
    @staticmethod
    async def config_exists(model_name: str) -> bool:
        """Check if a training configuration exists for a model"""
        try:
            query = "SELECT 1 FROM training_configurations WHERE model_name = $1 LIMIT 1"
            result = await db_manager.execute_fetchone(query, model_name)
            return result is not None
        except Exception as e:
            logger.error(f"❌ Failed to check if config exists for {model_name}: {e}")
            return False
