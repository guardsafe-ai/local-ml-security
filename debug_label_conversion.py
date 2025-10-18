#!/usr/bin/env python3
"""
Debug label conversion issue
"""

import json
import boto3
from botocore.exceptions import ClientError

def debug_label_conversion():
    """Debug the label conversion process"""
    
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
        
        print(f"📊 Loaded {len(lines)} lines from MinIO")
        
        # Parse data
        data = []
        for line in lines:
            data.append(json.loads(line))
        
        # Extract texts and labels
        texts = [item['text'] for item in data]
        labels = [item['label'] for item in data]
        
        print(f"📋 Original data: {len(texts)} texts, {len(labels)} labels")
        print(f"📋 Empty texts: {sum(1 for t in texts if not t or not t.strip())}")
        print(f"📋 Empty labels: {sum(1 for l in labels if not l or not str(l).strip())}")
        
        # Show first 10 labels
        print(f"📋 First 10 labels: {labels[:10]}")
        
        # Create label mapping
        unique_labels = sorted(list(set(labels)))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        
        print(f"📋 Unique labels: {unique_labels}")
        print(f"📋 Label mapping: {label_map}")
        
        # Convert labels
        converted_labels = []
        for i, label in enumerate(labels):
            if not label or not str(label).strip():
                print(f"❌ Empty label at index {i}: '{label}'")
                continue
            if label in label_map:
                converted_value = label_map[label]
                converted_labels.append(converted_value)
                if i < 10:  # Show first 10 conversions
                    print(f"✅ Converted '{label}' -> {converted_value}")
            else:
                print(f"❌ Unknown label '{label}' at index {i}")
        
        print(f"📋 Converted labels: {len(converted_labels)}")
        print(f"📋 First 10 converted: {converted_labels[:10]}")
        
        # Check for empty values in converted labels
        empty_converted = sum(1 for l in converted_labels if not l or not str(l).strip())
        print(f"📋 Empty converted labels: {empty_converted}")
        
        if empty_converted > 0:
            empty_indices = [i for i, label in enumerate(converted_labels) if not label or not str(label).strip()]
            print(f"❌ Empty converted label indices: {empty_indices[:10]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    debug_label_conversion()
