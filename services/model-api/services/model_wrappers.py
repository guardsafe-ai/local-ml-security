"""
Model Wrapper Classes
"""

import logging
import tempfile
import os
import torch
import mlflow
import joblib
from typing import Dict, List, Optional, Any, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class PyTorchModel:
    """Wrapper for PyTorch models with ONNX optimization"""
    
    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.labels = ["prompt_injection", "jailbreak", "system_extraction", "code_injection", "benign"]
        self.loaded = False
        self.model_source = "Unknown"
        self.model_version = "Unknown"
        # ONNX optimization
        self.onnx_session = None
        self.use_onnx = False
        self.onnx_path = None
        # Memory management
        self.temp_dirs = []  # Track temporary directories for cleanup
        self._cuda_initialized = torch.cuda.is_available()
    
    def load(self, enable_onnx: bool = True):
        """Load the model and tokenizer with optional ONNX optimization"""
        try:
            # Check if this is an MLflow model URI
            if self.model_path.startswith("models:/"):
                logger.info(f"🔄 [MLFLOW] Loading MLflow model: {self.model_path}")
                self._load_mlflow_model()
            else:
                logger.info(f"🔄 [LOCAL] Loading local model: {self.model_path}")
                self._load_local_model()
            
            # Convert to ONNX for faster inference
            if enable_onnx and self.model is not None:
                try:
                    self._convert_to_onnx()
                except Exception as e:
                    logger.warning(f"⚠️ ONNX conversion failed for {self.model_name}: {e}")
                    logger.info("🔄 Falling back to PyTorch inference")
                    self.use_onnx = False
            
            self.loaded = True
            logger.info(f"✅ Successfully loaded model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"❌ Error loading model {self.model_name}: {e}")
            raise
    
    def _load_mlflow_model(self):
        """Load model from MLflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download model from MLflow
            model_uri = mlflow.pytorch.load_model(self.model_path, dst_path=temp_dir)
            
            # Load model and tokenizer
            self.model = model_uri
            self.tokenizer = AutoTokenizer.from_pretrained(temp_dir)
            self.model_source = "MLflow"
            self.model_version = self.model_path.split("/")[-1] if "/" in self.model_path else "latest"
    
    def _load_local_model(self):
        """Load model from local path"""
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model_source = "Local"
        self.model_version = "local"
    
    def _convert_to_onnx(self):
        """Convert PyTorch model to ONNX for faster inference"""
        try:
            from services.onnx_converter import onnx_converter
            
            logger.info(f"🔄 [ONNX] Converting {self.model_name} to ONNX...")
            
            # Convert model to ONNX
            self.onnx_path = onnx_converter.convert_model_to_onnx(
                model=self.model,
                tokenizer=self.tokenizer,
                model_name=self.model_name,
                max_length=512
            )
            
            # Create ONNX Runtime session
            self.onnx_session = onnx_converter.create_onnx_session(
                onnx_path=self.onnx_path,
                model_name=self.model_name
            )
            
            self.use_onnx = True
            logger.info(f"✅ [ONNX] Successfully converted {self.model_name} to ONNX")
            
        except Exception as e:
            logger.error(f"❌ [ONNX] Conversion failed for {self.model_name}: {e}")
            raise
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Make prediction on text using ONNX if available, otherwise PyTorch"""
        if not self.loaded:
            raise ValueError(f"Model {self.model_name} is not loaded")
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Use ONNX if available, otherwise PyTorch
        if self.use_onnx and self.onnx_session is not None:
            try:
                from services.onnx_converter import onnx_converter
                result = onnx_converter.predict_with_onnx(
                    session=self.onnx_session,
                    input_ids=inputs['input_ids'],
                    model_name=self.model_name
                )
                return result
            except Exception as e:
                logger.warning(f"⚠️ ONNX prediction failed, falling back to PyTorch: {e}")
                self.use_onnx = False
        
        # PyTorch prediction (fallback or primary)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = torch.max(probabilities, dim=-1)[0].item()
        
        # Get probabilities for all classes
        prob_dict = {
            self.labels[i]: probabilities[0][i].item() 
            for i in range(len(self.labels))
        }
        
        return {
            "prediction": self.labels[predicted_class],
            "confidence": confidence,
            "probabilities": prob_dict,
            "inference_engine": "pytorch"
        }
    
    def unload(self):
        """Unload the model from memory with proper cleanup"""
        try:
            # Get memory info before unloading
            import psutil
            import os
            import gc
            import torch
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Clear model and tokenizer references
            self.model = None
            self.tokenizer = None
            self.loaded = False
            
            # Clear ONNX resources
            if self.onnx_session is not None:
                self.onnx_session = None
                self.use_onnx = False
            
            # Clean up ONNX file if it exists
            if self.onnx_path and os.path.exists(self.onnx_path):
                try:
                    os.remove(self.onnx_path)
                    logger.info(f"🗑️ Cleaned up ONNX file: {self.onnx_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to clean up ONNX file: {e}")
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Get memory info after unloading
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_freed = memory_before - memory_after
            
            logger.info(f"✅ Unloaded model: {self.model_name}")
            logger.info(f"💾 Memory freed: {memory_freed:.2f} MB")
            
        except Exception as e:
            logger.error(f"❌ Error during model unload: {e}")
            # Still clear references even if memory tracking fails
            self.model = None
            self.tokenizer = None
            self.loaded = False
            self.onnx_session = None
            self.use_onnx = False
    
    def cleanup_memory(self):
        """Clean up memory and temporary resources"""
        try:
            logger.info(f"🧹 [CLEANUP] Cleaning up memory for {self.model_name}")
            
            # Clear CUDA cache if available
            if self._cuda_initialized and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("🧹 [CLEANUP] CUDA cache cleared")
            
            # Clear model references
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            # Clear ONNX session
            if self.onnx_session is not None:
                del self.onnx_session
                self.onnx_session = None
            
            # Clean up temporary directories
            for temp_dir in self.temp_dirs:
                try:
                    if os.path.exists(temp_dir):
                        import shutil
                        shutil.rmtree(temp_dir)
                        logger.info(f"🧹 [CLEANUP] Removed temp directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"⚠️ [CLEANUP] Failed to remove temp directory {temp_dir}: {e}")
            
            self.temp_dirs.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info(f"✅ [CLEANUP] Memory cleanup completed for {self.model_name}")
            
        except Exception as e:
            logger.error(f"❌ [CLEANUP] Error during memory cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup_memory()
        except Exception as e:
            logger.debug(f"Error in destructor: {e}")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information"""
        try:
            memory_info = {
                "model_loaded": self.loaded,
                "model_size_mb": 0,
                "cuda_memory_allocated": 0,
                "cuda_memory_cached": 0,
                "temp_dirs_count": len(self.temp_dirs)
            }
            
            if self.model is not None:
                # Estimate model size
                try:
                    param_count = sum(p.numel() for p in self.model.parameters())
                    memory_info["model_size_mb"] = param_count * 4 / (1024 * 1024)  # Assuming float32
                except:
                    pass
            
            if self._cuda_initialized and torch.cuda.is_available():
                memory_info["cuda_memory_allocated"] = torch.cuda.memory_allocated() / (1024 * 1024)
                memory_info["cuda_memory_cached"] = torch.cuda.memory_reserved() / (1024 * 1024)
            
            return memory_info
            
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {"error": str(e)}


class SklearnModel:
    """Wrapper for Scikit-learn models"""
    
    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        self.vectorizer = None
        self.labels = ["prompt_injection", "jailbreak", "system_extraction", "code_injection", "benign"]
        self.loaded = False
        self.model_source = "Unknown"
        self.model_version = "Unknown"
    
    def load(self):
        """Load the model and vectorizer"""
        try:
            logger.info(f"🔄 [SKLEARN] Loading sklearn model: {self.model_path}")
            
            # Load model and vectorizer
            self.model = joblib.load(f"{self.model_path}/model.pkl")
            self.vectorizer = joblib.load(f"{self.model_path}/vectorizer.pkl")
            
            self.loaded = True
            self.model_source = "Sklearn"
            self.model_version = "local"
            logger.info(f"✅ Successfully loaded sklearn model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"❌ Error loading sklearn model {self.model_name}: {e}")
            raise
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Make prediction on text"""
        if not self.loaded:
            raise ValueError(f"Model {self.model_name} is not loaded")
        
        # Vectorize input
        text_vector = self.vectorizer.transform([text])
        
        # Make prediction
        prediction_proba = self.model.predict_proba(text_vector)[0]
        predicted_class_idx = prediction_proba.argmax()
        confidence = prediction_proba[predicted_class_idx]
        
        # Get probabilities for all classes
        prob_dict = {
            self.labels[i]: prediction_proba[i] 
            for i in range(len(self.labels))
        }
        
        return {
            "prediction": self.labels[predicted_class_idx],
            "confidence": confidence,
            "probabilities": prob_dict
        }
    
    def unload(self):
        """Unload the model from memory with proper cleanup"""
        try:
            # Get memory info before unloading
            import psutil
            import os
            import gc
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Clear model and vectorizer references
            self.model = None
            self.vectorizer = None
            self.loaded = False
            
            # Force garbage collection
            gc.collect()
            
            # Get memory info after unloading
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_freed = memory_before - memory_after
            
            logger.info(f"✅ Unloaded sklearn model: {self.model_name}")
            logger.info(f"💾 Memory freed: {memory_freed:.2f} MB")
            
        except Exception as e:
            logger.error(f"❌ Error during sklearn model unload: {e}")
            # Still clear references even if memory tracking fails
            self.model = None
            self.vectorizer = None
            self.loaded = False
    
    def cleanup_memory(self):
        """Clean up memory and resources"""
        try:
            logger.info(f"🧹 [CLEANUP] Cleaning up memory for sklearn model {self.model_name}")
            
            # Clear model references
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.vectorizer is not None:
                del self.vectorizer
                self.vectorizer = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info(f"✅ [CLEANUP] Memory cleanup completed for {self.model_name}")
            
        except Exception as e:
            logger.error(f"❌ [CLEANUP] Error during sklearn memory cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup_memory()
        except Exception as e:
            logger.debug(f"Error in sklearn destructor: {e}")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            return {
                "model_loaded": self.loaded,
                "process_memory_mb": memory_mb,
                "model_type": "sklearn"
            }
            
        except Exception as e:
            logger.error(f"Error getting sklearn memory usage: {e}")
            return {"error": str(e)}
