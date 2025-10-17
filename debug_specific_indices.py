#!/usr/bin/env python3
"""
Debug specific indices that are showing as empty
"""

import json
import boto3
from botocore.exceptions import ClientError

def debug_specific_indices():
    """Debug the specific indices that are showing as empty"""
    
    # Initialize MinIO client
    s3_client = boto3.client(
        's3',
        endpoint_url='http://localhost:9000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin',
        region_name='us-east-1'
    )
    
    try:
        # Load the data
        response = s3_client.get_object(
            Bucket='ml-security',
            Key='training-data/fresh/clean_training_data_20251017_213004.jsonl'
        )
        
        content = response['Body'].read().decode('utf-8')
        lines = content.strip().split('\n')
        
        # Parse data
        data = []
        for line in lines:
            data.append(json.loads(line))
        
        # Extract texts and labels
        texts = [item['text'] for item in data]
        labels = [item['label'] for item in data]
        
        # Create label mapping
        unique_labels = sorted(list(set(labels)))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Convert labels
        converted_labels = []
        for i, label in enumerate(labels):
            if label in label_map:
                converted_value = label_map[label]
                converted_labels.append(converted_value)
            else:
                print(f"❌ Unknown label '{label}' at index {i}")
                converted_labels.append(None)  # Use None for unknown labels
        
        # Check specific indices that were reported as empty
        problem_indices = [1, 3, 6, 12, 14, 20, 28, 33, 45, 49]
        
        print("🔍 Checking specific problem indices:")
        for idx in problem_indices:
            if idx < len(labels):
                original_label = labels[idx]
                converted_label = converted_labels[idx]
                print(f"Index {idx}: '{original_label}' -> {converted_label}")
                print(f"  - Original type: {type(original_label)}")
                print(f"  - Original repr: {repr(original_label)}")
                print(f"  - Converted type: {type(converted_label)}")
                print(f"  - Converted repr: {repr(converted_label)}")
                print(f"  - Is empty: {not converted_label or not str(converted_label).strip()}")
                print()
        
        # Check all converted labels for empty values
        empty_count = 0
        for i, label in enumerate(converted_labels):
            if not label or not str(label).strip():
                empty_count += 1
                if empty_count <= 10:  # Show first 10
                    print(f"Empty at index {i}: {repr(label)}")
        
        print(f"Total empty labels: {empty_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    debug_specific_indices()
