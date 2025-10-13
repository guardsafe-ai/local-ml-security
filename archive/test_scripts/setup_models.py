#!/usr/bin/env python3
"""
Setup script to download and configure pre-trained models
"""

import os
import sys
import requests
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def create_model_directory():
    """Create the models directory structure"""
    models_dir = Path("./data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created models directory: {models_dir}")

def download_model(model_name: str, hf_model_name: str):
    """Download a model from Hugging Face"""
    try:
        print(f"🔄 Downloading {model_name}...")
        
        # Create model directory
        model_dir = Path(f"./data/models/{model_name}_final")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Download tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_model_name,
            num_labels=5,  # 5 classes: prompt_injection, jailbreak, system_extraction, code_injection, benign
            problem_type="single_label_classification"
        )
        
        # Save model and tokenizer
        tokenizer.save_pretrained(str(model_dir))
        model.save_pretrained(str(model_dir))
        
        # Create config.json for the model
        config = {
            "model_type": "deberta" if "deberta" in model_name else "bert",
            "num_labels": 5,
            "id2label": {
                "0": "prompt_injection",
                "1": "jailbreak", 
                "2": "system_extraction",
                "3": "code_injection",
                "4": "benign"
            },
            "label2id": {
                "prompt_injection": 0,
                "jailbreak": 1,
                "system_extraction": 2,
                "code_injection": 3,
                "benign": 4
            }
        }
        
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Downloaded {model_name} to {model_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {model_name}: {e}")
        return False

def test_model_loading():
    """Test if models can be loaded by the API"""
    try:
        print("🔄 Testing model loading...")
        
        # Try to reload the model via API
        response = requests.post("http://localhost:8000/models/deberta-v3-large/reload")
        if response.status_code == 200:
            print("✅ Model API can load the model")
            return True
        else:
            print(f"❌ Model API failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing model loading: {e}")
        return False

def main():
    """Main function"""
    print("🤖 ML Security Model Setup")
    print("=" * 40)
    
    # Create directory structure
    create_model_directory()
    
    # Define models to download
    models = {
        "deberta-v3-large": "microsoft/deberta-v3-large",
        "roberta-large": "roberta-large",
        "bert-base": "bert-base-uncased",
        "distilbert": "distilbert-base-uncased"
    }
    
    print("\n🔄 Downloading models...")
    success_count = 0
    
    for model_name, hf_name in models.items():
        if download_model(model_name, hf_name):
            success_count += 1
        print()
    
    print(f"📊 Downloaded {success_count}/{len(models)} models successfully")
    
    # Test model loading
    print("\n🔄 Testing model loading...")
    if test_model_loading():
        print("✅ Models are ready to use!")
    else:
        print("⚠️  Models downloaded but API loading needs debugging")
    
    print("\n🎯 Next Steps:")
    print("1. Check the Streamlit dashboard: http://localhost:8501")
    print("2. Go to the 'Models' page to see loaded models")
    print("3. Use the 'Red Team' page to test model predictions")

if __name__ == "__main__":
    main()
