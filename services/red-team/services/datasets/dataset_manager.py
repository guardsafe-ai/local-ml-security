"""
Dataset Manager
Manages and coordinates all dataset generation and operations
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import asdict

from .owasp_dataset import OWASPDatasetGenerator, OWASPSample
from .nist_dataset import NISTDatasetGenerator, NISTSample
from .jailbreak_dataset import JailbreakDatasetGenerator, JailbreakSample
from .fairness_dataset import FairnessDatasetGenerator, FairnessSample
from .privacy_dataset import PrivacyDatasetGenerator, PrivacySample

logger = logging.getLogger(__name__)

class DatasetManager:
    """Manages all dataset generation and operations"""
    
    def __init__(self, output_dir: str = "./datasets"):
        self.output_dir = output_dir
        self.owasp_generator = OWASPDatasetGenerator()
        self.nist_generator = NISTDatasetGenerator()
        self.jailbreak_generator = JailbreakDatasetGenerator()
        self.fairness_generator = FairnessDatasetGenerator()
        self.privacy_generator = PrivacyDatasetGenerator()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Dataset statistics
        self.dataset_stats = {
            "owasp": {"total_samples": 0, "last_generated": None},
            "nist": {"total_samples": 0, "last_generated": None},
            "jailbreak": {"total_samples": 0, "last_generated": None},
            "fairness": {"total_samples": 0, "last_generated": None},
            "privacy": {"total_samples": 0, "last_generated": None}
        }
    
    def generate_owasp_dataset(self, samples_per_vulnerability: int = 100) -> List[OWASPSample]:
        """Generate OWASP LLM Top 10 dataset"""
        logger.info(f"Generating OWASP dataset with {samples_per_vulnerability} samples per vulnerability")
        
        samples = self.owasp_generator.generate_dataset(samples_per_vulnerability)
        
        # Update statistics
        self.dataset_stats["owasp"]["total_samples"] = len(samples)
        self.dataset_stats["owasp"]["last_generated"] = datetime.now().isoformat()
        
        # Export dataset
        filename = self.owasp_generator.export_dataset(samples)
        logger.info(f"OWASP dataset exported to {filename}")
        
        return samples
    
    def generate_nist_dataset(self, samples_per_category: int = 100) -> List[NISTSample]:
        """Generate NIST AI RMF dataset"""
        logger.info(f"Generating NIST dataset with {samples_per_category} samples per category")
        
        samples = self.nist_generator.generate_dataset(samples_per_category)
        
        # Update statistics
        self.dataset_stats["nist"]["total_samples"] = len(samples)
        self.dataset_stats["nist"]["last_generated"] = datetime.now().isoformat()
        
        # Export dataset
        filename = self.nist_generator.export_dataset(samples)
        logger.info(f"NIST dataset exported to {filename}")
        
        return samples
    
    def generate_jailbreak_dataset(self, samples_per_technique: int = 100) -> List[JailbreakSample]:
        """Generate jailbreak dataset"""
        logger.info(f"Generating jailbreak dataset with {samples_per_technique} samples per technique")
        
        samples = self.jailbreak_generator.generate_dataset(samples_per_technique)
        
        # Update statistics
        self.dataset_stats["jailbreak"]["total_samples"] = len(samples)
        self.dataset_stats["jailbreak"]["last_generated"] = datetime.now().isoformat()
        
        # Export dataset
        filename = self.jailbreak_generator.export_dataset(samples)
        logger.info(f"Jailbreak dataset exported to {filename}")
        
        return samples
    
    def generate_fairness_dataset(self, samples_per_attribute: int = 100) -> List[FairnessSample]:
        """Generate fairness dataset"""
        logger.info(f"Generating fairness dataset with {samples_per_attribute} samples per attribute")
        
        samples = self.fairness_generator.generate_dataset(samples_per_attribute)
        
        # Update statistics
        self.dataset_stats["fairness"]["total_samples"] = len(samples)
        self.dataset_stats["fairness"]["last_generated"] = datetime.now().isoformat()
        
        # Export dataset
        filename = self.fairness_generator.export_dataset(samples)
        logger.info(f"Fairness dataset exported to {filename}")
        
        return samples
    
    def generate_privacy_dataset(self, samples_per_risk: int = 100) -> List[PrivacySample]:
        """Generate privacy dataset"""
        logger.info(f"Generating privacy dataset with {samples_per_risk} samples per risk")
        
        samples = self.privacy_generator.generate_dataset(samples_per_risk)
        
        # Update statistics
        self.dataset_stats["privacy"]["total_samples"] = len(samples)
        self.dataset_stats["privacy"]["last_generated"] = datetime.now().isoformat()
        
        # Export dataset
        filename = self.privacy_generator.export_dataset(samples)
        logger.info(f"Privacy dataset exported to {filename}")
        
        return samples
    
    def generate_all_datasets(self, samples_per_category: int = 100) -> Dict[str, List]:
        """Generate all datasets"""
        logger.info("Generating all datasets...")
        
        datasets = {}
        
        try:
            datasets["owasp"] = self.generate_owasp_dataset(samples_per_category)
            datasets["nist"] = self.generate_nist_dataset(samples_per_category)
            datasets["jailbreak"] = self.generate_jailbreak_dataset(samples_per_category)
            datasets["fairness"] = self.generate_fairness_dataset(samples_per_category)
            datasets["privacy"] = self.generate_privacy_dataset(samples_per_category)
            
            logger.info("All datasets generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating datasets: {e}")
            raise
        
        return datasets
    
    def load_dataset(self, dataset_type: str, filename: str) -> List[Dict[str, Any]]:
        """Load dataset from file"""
        logger.info(f"Loading {dataset_type} dataset from {filename}")
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            samples = data.get("samples", [])
            logger.info(f"Loaded {len(samples)} samples from {filename}")
            
            return samples
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def save_dataset(self, samples: List[Dict[str, Any]], dataset_type: str, filename: str = None) -> str:
        """Save dataset to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{dataset_type}_dataset_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        data = {
            "dataset_type": dataset_type,
            "created_at": datetime.now().isoformat(),
            "total_samples": len(samples),
            "samples": samples
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Dataset saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            raise
    
    def merge_datasets(self, datasets: List[List[Dict[str, Any]]], dataset_type: str) -> List[Dict[str, Any]]:
        """Merge multiple datasets of the same type"""
        logger.info(f"Merging {len(datasets)} {dataset_type} datasets")
        
        merged_samples = []
        for dataset in datasets:
            merged_samples.extend(dataset)
        
        logger.info(f"Merged dataset contains {len(merged_samples)} samples")
        return merged_samples
    
    def split_dataset(self, samples: List[Dict[str, Any]], train_ratio: float = 0.7, 
                     val_ratio: float = 0.15, test_ratio: float = 0.15) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split dataset into train, validation, and test sets"""
        logger.info(f"Splitting dataset into train ({train_ratio}), val ({val_ratio}), test ({test_ratio})")
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        # Shuffle samples
        import random
        random.shuffle(samples)
        
        total_samples = len(samples)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        train_samples = samples[:train_size]
        val_samples = samples[train_size:train_size + val_size]
        test_samples = samples[train_size + val_size:]
        
        logger.info(f"Split: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")
        
        return train_samples, val_samples, test_samples
    
    def get_dataset_statistics(self, samples: List[Dict[str, Any]], dataset_type: str) -> Dict[str, Any]:
        """Get statistics for a dataset"""
        logger.info(f"Calculating statistics for {dataset_type} dataset")
        
        stats = {
            "total_samples": len(samples),
            "dataset_type": dataset_type,
            "created_at": datetime.now().isoformat()
        }
        
        if dataset_type == "owasp":
            # Count by vulnerability type
            vuln_counts = {}
            for sample in samples:
                vuln_type = sample.get("vulnerability_type", "unknown")
                vuln_counts[vuln_type] = vuln_counts.get(vuln_type, 0) + 1
            stats["vulnerability_counts"] = vuln_counts
            
        elif dataset_type == "nist":
            # Count by risk category
            risk_counts = {}
            for sample in samples:
                risk_category = sample.get("risk_category", "unknown")
                risk_counts[risk_category] = risk_counts.get(risk_category, 0) + 1
            stats["risk_category_counts"] = risk_counts
            
        elif dataset_type == "jailbreak":
            # Count by technique
            technique_counts = {}
            for sample in samples:
                technique = sample.get("technique", "unknown")
                technique_counts[technique] = technique_counts.get(technique, 0) + 1
            stats["technique_counts"] = technique_counts
            
        elif dataset_type == "fairness":
            # Count by protected attribute
            attribute_counts = {}
            for sample in samples:
                attribute = sample.get("protected_attribute", "unknown")
                attribute_counts[attribute] = attribute_counts.get(attribute, 0) + 1
            stats["protected_attribute_counts"] = attribute_counts
            
        elif dataset_type == "privacy":
            # Count by privacy risk
            risk_counts = {}
            for sample in samples:
                risk = sample.get("privacy_risk", "unknown")
                risk_counts[risk] = risk_counts.get(risk, 0) + 1
            stats["privacy_risk_counts"] = risk_counts
        
        return stats
    
    def export_dataset_statistics(self, stats: Dict[str, Any], filename: str = None) -> str:
        """Export dataset statistics to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dataset_statistics_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Dataset statistics exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting statistics: {e}")
            raise
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global statistics for all datasets"""
        logger.info("Calculating global dataset statistics")
        
        total_samples = sum(stats["total_samples"] for stats in self.dataset_stats.values())
        
        global_stats = {
            "total_datasets": len(self.dataset_stats),
            "total_samples": total_samples,
            "dataset_statistics": self.dataset_stats,
            "generated_at": datetime.now().isoformat()
        }
        
        return global_stats
    
    def export_global_statistics(self, filename: str = None) -> str:
        """Export global statistics to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"global_dataset_statistics_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        stats = self.get_global_statistics()
        
        try:
            with open(filepath, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Global statistics exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting global statistics: {e}")
            raise
    
    def validate_dataset(self, samples: List[Dict[str, Any]], dataset_type: str) -> Dict[str, Any]:
        """Validate dataset for completeness and consistency"""
        logger.info(f"Validating {dataset_type} dataset")
        
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "total_samples": len(samples)
        }
        
        required_fields = {
            "owasp": ["prompt", "expected_response", "vulnerability_type", "severity"],
            "nist": ["scenario", "risk_level", "risk_category", "trustworthiness_dimension"],
            "jailbreak": ["prompt", "expected_response", "jailbreak_type", "technique"],
            "fairness": ["scenario", "protected_attribute", "bias_type", "fairness_metric"],
            "privacy": ["scenario", "privacy_risk", "data_type", "privacy_metric"]
        }
        
        if dataset_type not in required_fields:
            validation_results["errors"].append(f"Unknown dataset type: {dataset_type}")
            validation_results["valid"] = False
            return validation_results
        
        required = required_fields[dataset_type]
        
        for i, sample in enumerate(samples):
            for field in required:
                if field not in sample:
                    validation_results["errors"].append(f"Sample {i}: Missing required field '{field}'")
                    validation_results["valid"] = False
                
                if sample.get(field) is None or sample.get(field) == "":
                    validation_results["warnings"].append(f"Sample {i}: Empty value for field '{field}'")
        
        logger.info(f"Validation complete: {len(validation_results['errors'])} errors, {len(validation_results['warnings'])} warnings")
        
        return validation_results
    
    def clean_dataset(self, samples: List[Dict[str, Any]], dataset_type: str) -> List[Dict[str, Any]]:
        """Clean dataset by removing invalid samples"""
        logger.info(f"Cleaning {dataset_type} dataset")
        
        required_fields = {
            "owasp": ["prompt", "expected_response", "vulnerability_type", "severity"],
            "nist": ["scenario", "risk_level", "risk_category", "trustworthiness_dimension"],
            "jailbreak": ["prompt", "expected_response", "jailbreak_type", "technique"],
            "fairness": ["scenario", "protected_attribute", "bias_type", "fairness_metric"],
            "privacy": ["scenario", "privacy_risk", "data_type", "privacy_metric"]
        }
        
        if dataset_type not in required_fields:
            logger.error(f"Unknown dataset type: {dataset_type}")
            return samples
        
        required = required_fields[dataset_type]
        cleaned_samples = []
        
        for sample in samples:
            # Check if all required fields are present and non-empty
            if all(field in sample and sample.get(field) is not None and sample.get(field) != "" for field in required):
                cleaned_samples.append(sample)
            else:
                logger.warning(f"Removing invalid sample: {sample}")
        
        logger.info(f"Cleaned dataset: {len(cleaned_samples)} samples remaining (removed {len(samples) - len(cleaned_samples)})")
        
        return cleaned_samples
    
    def get_dataset_info(self, dataset_type: str) -> Dict[str, Any]:
        """Get information about a specific dataset type"""
        if dataset_type == "owasp":
            return {
                "name": "OWASP LLM Top 10",
                "description": "Training dataset for OWASP LLM Top 10 vulnerabilities",
                "vulnerabilities": self.owasp_generator.get_all_vulnerabilities(),
                "total_samples": self.dataset_stats["owasp"]["total_samples"],
                "last_generated": self.dataset_stats["owasp"]["last_generated"]
            }
        elif dataset_type == "nist":
            return {
                "name": "NIST AI RMF",
                "description": "Training dataset for NIST AI Risk Management Framework",
                "risk_categories": self.nist_generator.get_all_risk_categories(),
                "total_samples": self.dataset_stats["nist"]["total_samples"],
                "last_generated": self.dataset_stats["nist"]["last_generated"]
            }
        elif dataset_type == "jailbreak":
            return {
                "name": "Jailbreak Attacks and Defenses",
                "description": "Training dataset for jailbreak attacks and defense strategies",
                "techniques": self.jailbreak_generator.get_all_techniques(),
                "total_samples": self.dataset_stats["jailbreak"]["total_samples"],
                "last_generated": self.dataset_stats["jailbreak"]["last_generated"]
            }
        elif dataset_type == "fairness":
            return {
                "name": "Fairness Testing and Bias Detection",
                "description": "Training dataset for fairness testing and bias detection",
                "protected_attributes": self.fairness_generator.get_all_protected_attributes(),
                "total_samples": self.dataset_stats["fairness"]["total_samples"],
                "last_generated": self.dataset_stats["fairness"]["last_generated"]
            }
        elif dataset_type == "privacy":
            return {
                "name": "Privacy Testing and Data Protection",
                "description": "Training dataset for privacy testing and data protection",
                "privacy_risks": self.privacy_generator.get_all_privacy_risks(),
                "total_samples": self.dataset_stats["privacy"]["total_samples"],
                "last_generated": self.dataset_stats["privacy"]["last_generated"]
            }
        else:
            return {"error": f"Unknown dataset type: {dataset_type}"}
    
    def list_available_datasets(self) -> List[str]:
        """List all available dataset types"""
        return list(self.dataset_stats.keys())
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get summary of all datasets"""
        summary = {
            "available_datasets": self.list_available_datasets(),
            "total_datasets": len(self.dataset_stats),
            "total_samples": sum(stats["total_samples"] for stats in self.dataset_stats.values()),
            "dataset_details": {}
        }
        
        for dataset_type in self.dataset_stats.keys():
            summary["dataset_details"][dataset_type] = self.get_dataset_info(dataset_type)
        
        return summary
