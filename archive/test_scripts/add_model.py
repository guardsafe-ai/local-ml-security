#!/usr/bin/env python3
"""
Simple script to add a single model to the ML Security Service
"""

import sys
import requests
import time
from pathlib import Path

def add_model(model_name: str, hf_model_name: str, model_type: str = "pytorch", priority: int = 5):
    """Add a single model to the system"""
    
    print(f"🤖 Adding model: {model_name}")
    print(f"   Hugging Face: {hf_model_name}")
    print(f"   Type: {model_type}")
    print(f"   Priority: {priority}")
    
    # Check if services are running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ Model API service is not running. Please start services first:")
            print("   docker-compose up -d")
            return False
    except:
        print("❌ Cannot connect to Model API service. Please start services first:")
        print("   docker-compose up -d")
        return False
    
    # Read current configuration
    model_api_path = Path("./services/model-api/main.py")
    if not model_api_path.exists():
        print(f"❌ Model API file not found: {model_api_path}")
        return False
    
    content = model_api_path.read_text()
    
    # Check if model already exists
    if f'"{model_name}"' in content:
        print(f"⚠️  Model '{model_name}' already exists in configuration")
        return True
    
    # Find the model_configs section
    start_marker = "self.model_configs = {"
    end_marker = "        }"
    
    start_idx = content.find(start_marker)
    if start_idx == -1:
        print("❌ Could not find model_configs section")
        return False
    
    # Find the end of the model_configs dictionary
    brace_count = 0
    end_idx = start_idx
    for i, char in enumerate(content[start_idx:], start_idx):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break
    
    if end_idx == start_idx:
        print("❌ Could not find end of model_configs section")
        return False
    
    # Create new model entry
    new_model_entry = f'''            "{model_name}": {{
                "type": "{model_type}",
                "path": "{hf_model_name}",
                "priority": {priority}
            }},'''
    
    # Insert new model before the closing brace
    current_config = content[start_idx:end_idx]
    new_config = current_config.replace(end_marker, f"{new_model_entry}\n{end_marker}")
    
    # Replace in content
    new_content = content[:start_idx] + new_config + content[end_idx:]
    
    # Write back to file
    model_api_path.write_text(new_content)
    print(f"✅ Added '{model_name}' to model-api configuration")
    
    print(f"\n🔄 Next steps:")
    print(f"1. Restart the model-api service:")
    print(f"   docker-compose restart model-api")
    print(f"\n2. Wait for the model to load (may take a few minutes)")
    print(f"\n3. Test the model:")
    print(f"   curl -X POST http://localhost:8000/predict \\")
    print(f"     -H 'Content-Type: application/json' \\")
    print(f"     -d '{{\"text\": \"Test text\", \"models\": [\"{model_name}_pretrained\"]}}'")
    print(f"\n4. Check the monitoring dashboard:")
    print(f"   http://localhost:8501")
    
    return True

def main():
    """Main function"""
    if len(sys.argv) < 3:
        print("Usage: python add_model.py <model_name> <huggingface_model_name> [type] [priority]")
        print("\nExamples:")
        print("  python add_model.py electra-base google/electra-base-discriminator")
        print("  python add_model.py albert-base albert-base-v2 pytorch 6")
        print("  python add_model.py custom-model ./path/to/local/model pytorch 7")
        sys.exit(1)
    
    model_name = sys.argv[1]
    hf_model_name = sys.argv[2]
    model_type = sys.argv[3] if len(sys.argv) > 3 else "pytorch"
    priority = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    
    success = add_model(model_name, hf_model_name, model_type, priority)
    
    if success:
        print(f"\n✅ Model '{model_name}' added successfully!")
    else:
        print(f"\n❌ Failed to add model '{model_name}'")
        sys.exit(1)

if __name__ == "__main__":
    main()
