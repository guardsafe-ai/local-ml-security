"""
ONNX Model Conversion Service
Converts PyTorch models to ONNX for faster inference
"""

import logging
import torch
import torch.onnx
import onnx
import onnxruntime as ort
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any, Optional, Tuple
import os
import tempfile

logger = logging.getLogger(__name__)

class ONNXConverter:
    """Handles conversion of PyTorch models to ONNX format"""
    
    def __init__(self):
        self.onnx_models = {}
        self.onnx_sessions = {}
        
    def convert_model_to_onnx(self, model: torch.nn.Module, 
                            tokenizer: Any,
                            model_name: str,
                            max_length: int = 512) -> str:
        """
        Convert PyTorch model to ONNX format
        
        Args:
            model: PyTorch model to convert
            tokenizer: Tokenizer for the model
            model_name: Name of the model
            max_length: Maximum sequence length
            
        Returns:
            Path to the converted ONNX model
        """
        try:
            logger.info(f"🔄 [ONNX] Converting {model_name} to ONNX format")
            
            # Set model to evaluation mode
            model.eval()
            
            # Create dummy input for tracing
            dummy_input = torch.randint(0, tokenizer.vocab_size, (1, max_length))
            
            # Create temporary file for ONNX model
            onnx_path = f"/tmp/{model_name}_model.onnx"
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input_ids'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'logits': {0: 'batch_size'}
                }
            )
            
            # Verify the ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            logger.info(f"✅ [ONNX] Successfully converted {model_name} to ONNX")
            return onnx_path
            
        except Exception as e:
            logger.error(f"❌ [ONNX] Failed to convert {model_name} to ONNX: {e}")
            raise
    
    def create_onnx_session(self, onnx_path: str, model_name: str) -> ort.InferenceSession:
        """
        Create ONNX Runtime inference session
        
        Args:
            onnx_path: Path to ONNX model file
            model_name: Name of the model
            
        Returns:
            ONNX Runtime inference session
        """
        try:
            # Configure ONNX Runtime providers
            providers = ['CPUExecutionProvider']
            if torch.cuda.is_available():
                providers.insert(0, 'CUDAExecutionProvider')
            
            # Create session options
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            
            # Create inference session
            session = ort.InferenceSession(
                onnx_path,
                sess_options=session_options,
                providers=providers
            )
            
            self.onnx_sessions[model_name] = session
            logger.info(f"✅ [ONNX] Created inference session for {model_name}")
            
            return session
            
        except Exception as e:
            logger.error(f"❌ [ONNX] Failed to create session for {model_name}: {e}")
            raise
    
    def predict_with_onnx(self, session: ort.InferenceSession,
                         input_ids: torch.Tensor,
                         model_name: str) -> Dict[str, Any]:
        """
        Make prediction using ONNX Runtime
        
        Args:
            session: ONNX Runtime inference session
            input_ids: Input token IDs
            model_name: Name of the model
            
        Returns:
            Prediction results
        """
        try:
            # Convert input to numpy
            input_numpy = input_ids.cpu().numpy()
            
            # Run inference
            outputs = session.run(None, {'input_ids': input_numpy})
            logits = outputs[0]
            
            # Convert to probabilities
            probabilities = torch.softmax(torch.tensor(logits), dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Map to labels
            labels = ["prompt_injection", "jailbreak", "system_extraction", "code_injection", "benign"]
            prediction = labels[predicted_class]
            
            # Create probability dictionary
            prob_dict = {
                labels[i]: probabilities[0][i].item() 
                for i in range(len(labels))
            }
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": prob_dict,
                "model_name": model_name,
                "inference_engine": "onnx"
            }
            
        except Exception as e:
            logger.error(f"❌ [ONNX] Prediction failed for {model_name}: {e}")
            raise
    
    def benchmark_onnx_vs_pytorch(self, pytorch_model: torch.nn.Module,
                                onnx_session: ort.InferenceSession,
                                tokenizer: Any,
                                test_texts: list,
                                model_name: str) -> Dict[str, Any]:
        """
        Benchmark ONNX vs PyTorch performance
        
        Args:
            pytorch_model: Original PyTorch model
            onnx_session: ONNX Runtime session
            tokenizer: Tokenizer
            test_texts: List of test texts
            model_name: Name of the model
            
        Returns:
            Benchmark results
        """
        try:
            import time
            
            # Prepare test data
            test_inputs = []
            for text in test_texts[:10]:  # Test with first 10 texts
                tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                test_inputs.append(tokens['input_ids'])
            
            # Benchmark PyTorch
            pytorch_times = []
            with torch.no_grad():
                for input_ids in test_inputs:
                    start_time = time.time()
                    outputs = pytorch_model(input_ids)
                    pytorch_times.append(time.time() - start_time)
            
            # Benchmark ONNX
            onnx_times = []
            for input_ids in test_inputs:
                start_time = time.time()
                self.predict_with_onnx(onnx_session, input_ids, model_name)
                onnx_times.append(time.time() - start_time)
            
            # Calculate statistics
            pytorch_avg = sum(pytorch_times) / len(pytorch_times)
            onnx_avg = sum(onnx_times) / len(onnx_times)
            speedup = pytorch_avg / onnx_avg if onnx_avg > 0 else 0
            
            results = {
                "model_name": model_name,
                "pytorch_avg_time": pytorch_avg,
                "onnx_avg_time": onnx_avg,
                "speedup_factor": speedup,
                "test_samples": len(test_inputs),
                "pytorch_times": pytorch_times,
                "onnx_times": onnx_times
            }
            
            logger.info(f"📊 [ONNX] Benchmark for {model_name}: {speedup:.2f}x speedup")
            return results
            
        except Exception as e:
            logger.error(f"❌ [ONNX] Benchmark failed for {model_name}: {e}")
            return {"error": str(e)}
    
    def get_onnx_model_info(self, onnx_path: str) -> Dict[str, Any]:
        """
        Get information about ONNX model
        
        Args:
            onnx_path: Path to ONNX model file
            
        Returns:
            Model information
        """
        try:
            model = onnx.load(onnx_path)
            
            # Get model size
            model_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            # Get input/output info
            input_info = []
            output_info = []
            
            for input_tensor in model.graph.input:
                input_info.append({
                    "name": input_tensor.name,
                    "type": str(input_tensor.type),
                    "shape": [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
                })
            
            for output_tensor in model.graph.output:
                output_info.append({
                    "name": output_tensor.name,
                    "type": str(output_tensor.type),
                    "shape": [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
                })
            
            return {
                "model_size_mb": model_size,
                "opset_version": model.opset_import[0].version,
                "inputs": input_info,
                "outputs": output_info,
                "nodes_count": len(model.graph.node)
            }
            
        except Exception as e:
            logger.error(f"❌ [ONNX] Failed to get model info: {e}")
            return {"error": str(e)}

# Global ONNX converter instance
onnx_converter = ONNXConverter()
