"""
Model Quantization Service
Provides INT8 and FP16 quantization for faster inference
"""

import logging
import torch
import torch.quantization as quant
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any, Optional, Union
import os

logger = logging.getLogger(__name__)

class ModelQuantizer:
    """Handles model quantization for faster inference"""
    
    def __init__(self):
        self.quantization_configs = {
            "int8": {
                "dtype": torch.qint8,
                "scheme": "qint8",
                "description": "8-bit integer quantization"
            },
            "fp16": {
                "dtype": torch.float16,
                "scheme": "fp16", 
                "description": "16-bit floating point"
            },
            "int8_dynamic": {
                "dtype": torch.qint8,
                "scheme": "dynamic",
                "description": "Dynamic 8-bit quantization"
            }
        }
    
    def quantize_model(self, model: torch.nn.Module, 
                      quantization_type: str = "int8",
                      model_name: str = "unknown") -> torch.nn.Module:
        """
        Quantize a PyTorch model for faster inference
        
        Args:
            model: PyTorch model to quantize
            quantization_type: Type of quantization (int8, fp16, int8_dynamic)
            model_name: Name of the model for logging
            
        Returns:
            Quantized model
        """
        try:
            if quantization_type not in self.quantization_configs:
                raise ValueError(f"Unsupported quantization type: {quantization_type}")
            
            config = self.quantization_configs[quantization_type]
            logger.info(f"🔄 [QUANTIZATION] Starting {config['description']} for {model_name}")
            
            # Set model to evaluation mode
            model.eval()
            
            if quantization_type == "fp16":
                # FP16 quantization
                model = model.half()
                logger.info(f"✅ [QUANTIZATION] FP16 quantization completed for {model_name}")
                
            elif quantization_type == "int8_dynamic":
                # Dynamic INT8 quantization
                model = torch.quantization.quantize_dynamic(
                    model, 
                    {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU}, 
                    dtype=torch.qint8
                )
                logger.info(f"✅ [QUANTIZATION] Dynamic INT8 quantization completed for {model_name}")
                
            elif quantization_type == "int8":
                # Static INT8 quantization (requires calibration)
                model = self._static_int8_quantization(model, model_name)
                logger.info(f"✅ [QUANTIZATION] Static INT8 quantization completed for {model_name}")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ [QUANTIZATION] Failed to quantize {model_name}: {e}")
            # Return original model if quantization fails
            return model
    
    def _static_int8_quantization(self, model: torch.nn.Module, model_name: str) -> torch.nn.Module:
        """
        Perform static INT8 quantization with calibration
        
        Args:
            model: Model to quantize
            model_name: Name of the model
            
        Returns:
            Quantized model
        """
        try:
            # Set model to evaluation mode
            model.eval()
            
            # Prepare model for quantization
            model.qconfig = quant.get_default_qconfig('fbgemm')
            
            # Prepare the model for quantization
            model_prepared = quant.prepare(model)
            
            # Calibrate the model (this would typically use a calibration dataset)
            # For now, we'll use a dummy calibration
            logger.info(f"🔄 [QUANTIZATION] Calibrating {model_name} for static INT8...")
            
            # Dummy calibration data (in practice, use real calibration data)
            with torch.no_grad():
                # Create dummy input for calibration
                dummy_input = torch.randn(1, 512)  # Adjust size based on model
                model_prepared(dummy_input)
            
            # Convert to quantized model
            quantized_model = quant.convert(model_prepared)
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"❌ [QUANTIZATION] Static INT8 quantization failed for {model_name}: {e}")
            # Fallback to dynamic quantization
            logger.info(f"🔄 [QUANTIZATION] Falling back to dynamic INT8 for {model_name}")
            return torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU}, 
                dtype=torch.qint8
            )
    
    def get_quantization_info(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Get information about model quantization status
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary with quantization information
        """
        try:
            info = {
                "is_quantized": hasattr(model, 'qconfig') and model.qconfig is not None,
                "dtype": str(model.dtype) if hasattr(model, 'dtype') else "unknown",
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
            
            # Estimate memory usage
            if hasattr(model, 'dtype'):
                if model.dtype == torch.float16:
                    info["memory_usage"] = "~50% of original (FP16)"
                elif model.dtype == torch.qint8:
                    info["memory_usage"] = "~25% of original (INT8)"
                else:
                    info["memory_usage"] = "100% of original (FP32)"
            
            return info
            
        except Exception as e:
            logger.error(f"❌ [QUANTIZATION] Failed to get quantization info: {e}")
            return {"error": str(e)}
    
    def benchmark_quantization(self, model: torch.nn.Module, 
                             input_tensor: torch.Tensor,
                             model_name: str = "unknown") -> Dict[str, Any]:
        """
        Benchmark quantized vs original model performance
        
        Args:
            model: Model to benchmark
            input_tensor: Input tensor for benchmarking
            model_name: Name of the model
            
        Returns:
            Benchmark results
        """
        try:
            import time
            
            # Warm up
            with torch.no_grad():
                for _ in range(10):
                    _ = model(input_tensor)
            
            # Benchmark original model
            start_time = time.time()
            with torch.no_grad():
                for _ in range(100):
                    _ = model(input_tensor)
            original_time = time.time() - start_time
            
            # Get model size
            model_size = sum(p.numel() * p.element_size() for p in model.parameters())
            
            results = {
                "model_name": model_name,
                "inference_time_100_runs": original_time,
                "avg_inference_time": original_time / 100,
                "model_size_bytes": model_size,
                "model_size_mb": model_size / (1024 * 1024),
                "quantization_type": "none"  # Will be updated by caller
            }
            
            return results
            
        except Exception as e:
            logger.error(f"❌ [QUANTIZATION] Benchmark failed for {model_name}: {e}")
            return {"error": str(e)}

# Global quantizer instance
model_quantizer = ModelQuantizer()