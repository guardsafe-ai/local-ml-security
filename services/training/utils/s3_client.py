"""
Training Service - S3/MinIO Client
Client for interacting with MinIO storage
"""

import boto3
import json
import logging
from typing import List, Dict, Any, Tuple
from io import StringIO
from utils.config import get_config

logger = logging.getLogger(__name__)


class S3Client:
    """S3/MinIO client for data operations"""
    
    def __init__(self):
        config = get_config()
        self.s3_client = boto3.client(
            's3',
            endpoint_url=config['minio_endpoint'],
            aws_access_key_id=config['aws_access_key_id'],
            aws_secret_access_key=config['aws_secret_access_key']
        )
        self.bucket_name = 'ml-security'
        
    def parse_s3_path(self, s3_path: str) -> Tuple[str, str]:
        """Parse S3 path to extract bucket and key"""
        if s3_path.startswith('s3://'):
            path_parts = s3_path[5:].split('/', 1)
            bucket = path_parts[0]
            key = path_parts[1] if len(path_parts) > 1 else ''
        else:
            # Assume it's just a key, use default bucket
            bucket = self.bucket_name
            key = s3_path
        return bucket, key
    
    def load_jsonl_data(self, s3_path: str) -> List[Dict[str, Any]]:
        """Load JSONL data from S3/MinIO"""
        try:
            bucket, key = self.parse_s3_path(s3_path)
            
            # Get object from S3
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            
            # Parse JSONL
            data = []
            for line in content.strip().split('\n'):
                if line.strip():
                    data.append(json.loads(line))
            
            logger.info(f"Loaded {len(data)} records from s3://{bucket}/{key}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load JSONL data from {s3_path}: {e}")
            raise
    
    def load_csv_data(self, s3_path: str) -> List[Dict[str, Any]]:
        """Load CSV data from S3/MinIO"""
        try:
            import pandas as pd
            
            bucket, key = self.parse_s3_path(s3_path)
            
            # Get object from S3
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            
            # Parse CSV
            df = pd.read_csv(StringIO(content))
            data = df.to_dict('records')
            
            logger.info(f"Loaded {len(data)} records from s3://{bucket}/{key}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load CSV data from {s3_path}: {e}")
            raise
    
    def upload_data(self, data: List[Dict[str, Any]], s3_path: str, format: str = 'jsonl') -> str:
        """Upload data to S3/MinIO"""
        try:
            bucket, key = self.parse_s3_path(s3_path)
            
            if format == 'jsonl':
                content = '\n'.join(json.dumps(item) for item in data)
            elif format == 'csv':
                import pandas as pd
                df = pd.DataFrame(data)
                content = df.to_csv(index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=content.encode('utf-8'),
                ContentType='application/json' if format == 'jsonl' else 'text/csv'
            )
            
            full_path = f"s3://{bucket}/{key}"
            logger.info(f"Uploaded {len(data)} records to {full_path}")
            return full_path
            
        except Exception as e:
            logger.error(f"Failed to upload data to {s3_path}: {e}")
            raise
    
    def list_objects(self, prefix: str = '') -> List[str]:
        """List objects in the bucket with given prefix"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            objects = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    objects.append(f"s3://{self.bucket_name}/{obj['Key']}")
            
            return objects
            
        except Exception as e:
            logger.error(f"Failed to list objects with prefix {prefix}: {e}")
            return []
    
    def object_exists(self, s3_path: str) -> bool:
        """Check if object exists in S3/MinIO"""
        try:
            bucket, key = self.parse_s3_path(s3_path)
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except:
            return False
