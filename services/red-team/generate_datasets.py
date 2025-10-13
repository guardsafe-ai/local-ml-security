#!/usr/bin/env python3
"""
Dataset Generation Script
Generates comprehensive training datasets for red team operations
"""

import argparse
import logging
import sys
import os
from typing import List, Dict, Any

# Add the services directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))

from services.datasets import DatasetManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate training datasets for red team operations")
    parser.add_argument(
        "--dataset-type",
        choices=["owasp", "nist", "jailbreak", "fairness", "privacy", "all"],
        default="all",
        help="Type of dataset to generate"
    )
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=100,
        help="Number of samples per category/vulnerability/technique"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./datasets",
        help="Output directory for generated datasets"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated datasets"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean generated datasets"
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split datasets into train/validation/test sets"
    )
    parser.add_argument(
        "--statistics",
        action="store_true",
        help="Generate dataset statistics"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize dataset manager
    manager = DatasetManager(output_dir=args.output_dir)
    
    try:
        if args.dataset_type == "all":
            logger.info("Generating all datasets...")
            datasets = manager.generate_all_datasets(args.samples_per_category)
            
            # Generate statistics for all datasets
            if args.statistics:
                logger.info("Generating statistics for all datasets...")
                for dataset_type, samples in datasets.items():
                    stats = manager.get_dataset_statistics(samples, dataset_type)
                    manager.export_dataset_statistics(stats, f"{dataset_type}_statistics.json")
                
                # Export global statistics
                manager.export_global_statistics()
            
            # Validate all datasets
            if args.validate:
                logger.info("Validating all datasets...")
                for dataset_type, samples in datasets.items():
                    validation_results = manager.validate_dataset(samples, dataset_type)
                    if not validation_results["valid"]:
                        logger.error(f"Validation failed for {dataset_type}: {validation_results['errors']}")
                    else:
                        logger.info(f"Validation passed for {dataset_type}")
            
            # Clean all datasets
            if args.clean:
                logger.info("Cleaning all datasets...")
                for dataset_type, samples in datasets.items():
                    cleaned_samples = manager.clean_dataset(samples, dataset_type)
                    if len(cleaned_samples) != len(samples):
                        logger.info(f"Cleaned {dataset_type}: {len(cleaned_samples)} samples remaining")
                        datasets[dataset_type] = cleaned_samples
            
            # Split all datasets
            if args.split:
                logger.info("Splitting all datasets...")
                for dataset_type, samples in datasets.items():
                    train, val, test = manager.split_dataset(samples)
                    
                    # Save split datasets
                    manager.save_dataset(train, f"{dataset_type}_train")
                    manager.save_dataset(val, f"{dataset_type}_val")
                    manager.save_dataset(test, f"{dataset_type}_test")
                    
                    logger.info(f"Split {dataset_type}: {len(train)} train, {len(val)} val, {len(test)} test")
            
            logger.info("All datasets generated successfully!")
            
        else:
            logger.info(f"Generating {args.dataset_type} dataset...")
            
            if args.dataset_type == "owasp":
                samples = manager.generate_owasp_dataset(args.samples_per_category)
            elif args.dataset_type == "nist":
                samples = manager.generate_nist_dataset(args.samples_per_category)
            elif args.dataset_type == "jailbreak":
                samples = manager.generate_jailbreak_dataset(args.samples_per_category)
            elif args.dataset_type == "fairness":
                samples = manager.generate_fairness_dataset(args.samples_per_category)
            elif args.dataset_type == "privacy":
                samples = manager.generate_privacy_dataset(args.samples_per_category)
            else:
                logger.error(f"Unknown dataset type: {args.dataset_type}")
                sys.exit(1)
            
            # Generate statistics
            if args.statistics:
                logger.info(f"Generating statistics for {args.dataset_type} dataset...")
                stats = manager.get_dataset_statistics(samples, args.dataset_type)
                manager.export_dataset_statistics(stats, f"{args.dataset_type}_statistics.json")
            
            # Validate dataset
            if args.validate:
                logger.info(f"Validating {args.dataset_type} dataset...")
                validation_results = manager.validate_dataset(samples, args.dataset_type)
                if not validation_results["valid"]:
                    logger.error(f"Validation failed: {validation_results['errors']}")
                else:
                    logger.info("Validation passed")
            
            # Clean dataset
            if args.clean:
                logger.info(f"Cleaning {args.dataset_type} dataset...")
                cleaned_samples = manager.clean_dataset(samples, args.dataset_type)
                if len(cleaned_samples) != len(samples):
                    logger.info(f"Cleaned: {len(cleaned_samples)} samples remaining")
                    samples = cleaned_samples
            
            # Split dataset
            if args.split:
                logger.info(f"Splitting {args.dataset_type} dataset...")
                train, val, test = manager.split_dataset(samples)
                
                # Save split datasets
                manager.save_dataset(train, f"{args.dataset_type}_train")
                manager.save_dataset(val, f"{args.dataset_type}_val")
                manager.save_dataset(test, f"{args.dataset_type}_test")
                
                logger.info(f"Split: {len(train)} train, {len(val)} val, {len(test)} test")
            
            logger.info(f"{args.dataset_type} dataset generated successfully!")
        
        # Print summary
        summary = manager.get_dataset_summary()
        logger.info(f"Dataset summary: {summary}")
        
    except Exception as e:
        logger.error(f"Error generating datasets: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
