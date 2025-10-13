"""
Fairness Dataset Generator
Generates training datasets for fairness testing and bias detection
"""

import json
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class FairnessSample:
    """Fairness sample data structure"""
    scenario: str
    protected_attribute: str
    bias_type: str
    fairness_metric: str
    expected_fairness_score: float
    description: str
    mitigation_strategy: str
    metadata: Dict[str, Any] = None

class FairnessDatasetGenerator:
    """Generates fairness testing and bias detection datasets"""
    
    def __init__(self):
        self.protected_attributes = {
            "gender": {
                "description": "Gender-based bias and discrimination",
                "values": ["male", "female", "non-binary", "transgender", "other"],
                "bias_examples": [
                    "Women are less likely to be hired for technical roles",
                    "Men are more likely to receive higher salaries",
                    "Non-binary individuals face discrimination in hiring",
                    "Transgender people are excluded from certain opportunities"
                ]
            },
            "race": {
                "description": "Racial bias and discrimination",
                "values": ["white", "black", "hispanic", "asian", "native_american", "other"],
                "bias_examples": [
                    "Black individuals are more likely to be denied loans",
                    "Hispanic people face discrimination in housing",
                    "Asian individuals are stereotyped in academic settings",
                    "Native Americans are underrepresented in media"
                ]
            },
            "age": {
                "description": "Age-based bias and discrimination",
                "values": ["young", "middle_aged", "elderly"],
                "bias_examples": [
                    "Older workers are less likely to be hired",
                    "Young people are seen as less experienced",
                    "Elderly individuals face age discrimination",
                    "Middle-aged workers are overlooked for promotions"
                ]
            },
            "disability": {
                "description": "Disability-based bias and discrimination",
                "values": ["disabled", "non_disabled"],
                "bias_examples": [
                    "Disabled individuals are less likely to be hired",
                    "People with disabilities face accessibility barriers",
                    "Disabled workers are paid less than non-disabled workers",
                    "People with disabilities are excluded from certain activities"
                ]
            },
            "religion": {
                "description": "Religious bias and discrimination",
                "values": ["christian", "muslim", "jewish", "hindu", "buddhist", "atheist", "other"],
                "bias_examples": [
                    "Muslim individuals face discrimination in hiring",
                    "Jewish people are stereotyped in financial contexts",
                    "Hindu individuals face bias in academic settings",
                    "Atheists are discriminated against in religious organizations"
                ]
            },
            "sexual_orientation": {
                "description": "Sexual orientation bias and discrimination",
                "values": ["heterosexual", "homosexual", "bisexual", "pansexual", "asexual", "other"],
                "bias_examples": [
                    "LGBTQ+ individuals face discrimination in hiring",
                    "Gay people are stereotyped in certain professions",
                    "Bisexual individuals face erasure and bias",
                    "Transgender people face discrimination in healthcare"
                ]
            },
            "socioeconomic_status": {
                "description": "Socioeconomic bias and discrimination",
                "values": ["low_income", "middle_income", "high_income"],
                "bias_examples": [
                    "Low-income individuals are less likely to receive loans",
                    "Poor people face discrimination in housing",
                    "Wealthy individuals receive preferential treatment",
                    "Middle-class people are overlooked in certain contexts"
                ]
            }
        }
        
        self.bias_types = {
            "statistical_parity": {
                "description": "Equal positive outcomes across groups",
                "formula": "P(Y=1|A=a) = P(Y=1|A=b) for all groups a, b",
                "examples": [
                    "Equal hiring rates across demographic groups",
                    "Equal loan approval rates across racial groups",
                    "Equal promotion rates across gender groups"
                ]
            },
            "equalized_odds": {
                "description": "Equal true positive and false positive rates across groups",
                "formula": "P(Y=1|A=a, Y_true=1) = P(Y=1|A=b, Y_true=1) and P(Y=1|A=a, Y_true=0) = P(Y=1|A=b, Y_true=0)",
                "examples": [
                    "Equal accuracy across demographic groups",
                    "Equal error rates across protected attributes",
                    "Equal performance across different groups"
                ]
            },
            "demographic_parity": {
                "description": "Equal selection rates across groups",
                "formula": "P(Y=1|A=a) = P(Y=1|A=b) for all groups a, b",
                "examples": [
                    "Equal selection rates in hiring",
                    "Equal admission rates in education",
                    "Equal approval rates in lending"
                ]
            },
            "individual_fairness": {
                "description": "Similar individuals receive similar outcomes",
                "formula": "If d(x1, x2) < δ, then |f(x1) - f(x2)| < ε",
                "examples": [
                    "Similar candidates receive similar scores",
                    "Similar applicants receive similar treatment",
                    "Similar individuals receive similar outcomes"
                ]
            },
            "counterfactual_fairness": {
                "description": "Outcome should be the same in counterfactual world",
                "formula": "P(Y|X, A) = P(Y|X, A') where A' is counterfactual protected attribute",
                "examples": [
                    "Outcome should not depend on protected attribute",
                    "Decision should be same regardless of protected attribute",
                    "Result should be independent of sensitive attribute"
                ]
            }
        }
        
        self.fairness_metrics = {
            "demographic_parity_difference": {
                "description": "Difference in positive outcome rates between groups",
                "range": [0, 1],
                "ideal_value": 0
            },
            "equalized_odds_difference": {
                "description": "Difference in true positive rates between groups",
                "range": [0, 1],
                "ideal_value": 0
            },
            "statistical_parity_difference": {
                "description": "Difference in selection rates between groups",
                "range": [0, 1],
                "ideal_value": 0
            },
            "calibration": {
                "description": "Predicted probabilities match actual outcomes",
                "range": [0, 1],
                "ideal_value": 1
            },
            "individual_fairness_score": {
                "description": "Consistency of outcomes for similar individuals",
                "range": [0, 1],
                "ideal_value": 1
            }
        }
        
        self.mitigation_strategies = {
            "preprocessing": [
                "Remove protected attributes from training data",
                "Use fair representation learning",
                "Apply data augmentation techniques",
                "Balance training data across groups",
                "Use synthetic data generation"
            ],
            "inprocessing": [
                "Add fairness constraints to model training",
                "Use adversarial debiasing",
                "Apply regularization techniques",
                "Use fair representation learning",
                "Implement fairness-aware optimization"
            ],
            "postprocessing": [
                "Adjust model outputs for fairness",
                "Use threshold optimization",
                "Apply calibration techniques",
                "Use outcome modification",
                "Implement fairness-aware decision making"
            ]
        }
    
    def generate_gender_bias_samples(self, count: int = 100) -> List[FairnessSample]:
        """Generate gender bias samples"""
        samples = []
        
        scenarios = [
            "AI system used for {application} shows bias against {gender}",
            "Model trained on {data} discriminates against {gender}",
            "Algorithm used for {purpose} treats {gender} unfairly",
            "AI system for {application} perpetuates bias against {gender}",
            "Model in {domain} shows discriminatory behavior toward {gender}",
            "AI system used for {purpose} exhibits bias in {context}",
            "Algorithm for {application} shows unfair treatment of {gender}",
            "Model in {domain} demonstrates bias against {gender}",
            "AI system for {purpose} discriminates in {context}",
            "Algorithm used for {application} shows bias toward {gender}"
        ]
        
        applications = [
            "hiring", "lending", "healthcare", "criminal justice", "education",
            "housing", "insurance", "recruitment", "admissions", "sentencing"
        ]
        
        genders = [
            "women", "men", "non-binary individuals", "transgender people", "gender minorities"
        ]
        
        data_types = [
            "historical data", "biased datasets", "incomplete data", "unrepresentative data"
        ]
        
        purposes = [
            "decision making", "risk assessment", "resource allocation", "evaluation"
        ]
        
        domains = [
            "finance", "healthcare", "education", "criminal justice", "employment"
        ]
        
        contexts = [
            "decision making", "risk assessment", "resource allocation", "evaluation"
        ]
        
        for _ in range(count):
            scenario = random.choice(scenarios)
            application = random.choice(applications)
            gender = random.choice(genders)
            data = random.choice(data_types)
            purpose = random.choice(purposes)
            domain = random.choice(domains)
            context = random.choice(contexts)
            
            scenario = scenario.format(
                application=application, gender=gender, data=data,
                purpose=purpose, domain=domain, context=context
            )
            
            sample = FairnessSample(
                scenario=scenario,
                protected_attribute="gender",
                bias_type=random.choice(list(self.bias_types.keys())),
                fairness_metric=random.choice(list(self.fairness_metrics.keys())),
                expected_fairness_score=random.uniform(0.3, 0.9),
                description=f"Gender bias in {application} affecting {gender}",
                mitigation_strategy=random.choice(random.choice(list(self.mitigation_strategies.values()))),
                metadata={
                    "application": application, "gender": gender, "data_type": data,
                    "purpose": purpose, "domain": domain, "context": context
                }
            )
            
            samples.append(sample)
        
        return samples
    
    def generate_racial_bias_samples(self, count: int = 100) -> List[FairnessSample]:
        """Generate racial bias samples"""
        samples = []
        
        scenarios = [
            "AI system used for {application} shows bias against {race}",
            "Model trained on {data} discriminates against {race}",
            "Algorithm used for {purpose} treats {race} unfairly",
            "AI system for {application} perpetuates bias against {race}",
            "Model in {domain} shows discriminatory behavior toward {race}",
            "AI system used for {purpose} exhibits bias in {context}",
            "Algorithm for {application} shows unfair treatment of {race}",
            "Model in {domain} demonstrates bias against {race}",
            "AI system for {purpose} discriminates in {context}",
            "Algorithm used for {application} shows bias toward {race}"
        ]
        
        applications = [
            "hiring", "lending", "healthcare", "criminal justice", "education",
            "housing", "insurance", "recruitment", "admissions", "sentencing"
        ]
        
        races = [
            "black individuals", "hispanic people", "asian individuals", "native americans",
            "racial minorities", "people of color", "non-white individuals"
        ]
        
        data_types = [
            "historical data", "biased datasets", "incomplete data", "unrepresentative data"
        ]
        
        purposes = [
            "decision making", "risk assessment", "resource allocation", "evaluation"
        ]
        
        domains = [
            "finance", "healthcare", "education", "criminal justice", "employment"
        ]
        
        contexts = [
            "decision making", "risk assessment", "resource allocation", "evaluation"
        ]
        
        for _ in range(count):
            scenario = random.choice(scenarios)
            application = random.choice(applications)
            race = random.choice(races)
            data = random.choice(data_types)
            purpose = random.choice(purposes)
            domain = random.choice(domains)
            context = random.choice(contexts)
            
            scenario = scenario.format(
                application=application, race=race, data=data,
                purpose=purpose, domain=domain, context=context
            )
            
            sample = FairnessSample(
                scenario=scenario,
                protected_attribute="race",
                bias_type=random.choice(list(self.bias_types.keys())),
                fairness_metric=random.choice(list(self.fairness_metrics.keys())),
                expected_fairness_score=random.uniform(0.3, 0.9),
                description=f"Racial bias in {application} affecting {race}",
                mitigation_strategy=random.choice(random.choice(list(self.mitigation_strategies.values()))),
                metadata={
                    "application": application, "race": race, "data_type": data,
                    "purpose": purpose, "domain": domain, "context": context
                }
            )
            
            samples.append(sample)
        
        return samples
    
    def generate_age_bias_samples(self, count: int = 100) -> List[FairnessSample]:
        """Generate age bias samples"""
        samples = []
        
        scenarios = [
            "AI system used for {application} shows bias against {age}",
            "Model trained on {data} discriminates against {age}",
            "Algorithm used for {purpose} treats {age} unfairly",
            "AI system for {application} perpetuates bias against {age}",
            "Model in {domain} shows discriminatory behavior toward {age}",
            "AI system used for {purpose} exhibits bias in {context}",
            "Algorithm for {application} shows unfair treatment of {age}",
            "Model in {domain} demonstrates bias against {age}",
            "AI system for {purpose} discriminates in {context}",
            "Algorithm used for {application} shows bias toward {age}"
        ]
        
        applications = [
            "hiring", "lending", "healthcare", "criminal justice", "education",
            "housing", "insurance", "recruitment", "admissions", "sentencing"
        ]
        
        age_groups = [
            "young people", "older workers", "elderly individuals", "middle-aged workers",
            "age minorities", "senior citizens", "young adults"
        ]
        
        data_types = [
            "historical data", "biased datasets", "incomplete data", "unrepresentative data"
        ]
        
        purposes = [
            "decision making", "risk assessment", "resource allocation", "evaluation"
        ]
        
        domains = [
            "finance", "healthcare", "education", "criminal justice", "employment"
        ]
        
        contexts = [
            "decision making", "risk assessment", "resource allocation", "evaluation"
        ]
        
        for _ in range(count):
            scenario = random.choice(scenarios)
            application = random.choice(applications)
            age = random.choice(age_groups)
            data = random.choice(data_types)
            purpose = random.choice(purposes)
            domain = random.choice(domains)
            context = random.choice(contexts)
            
            scenario = scenario.format(
                application=application, age=age, data=data,
                purpose=purpose, domain=domain, context=context
            )
            
            sample = FairnessSample(
                scenario=scenario,
                protected_attribute="age",
                bias_type=random.choice(list(self.bias_types.keys())),
                fairness_metric=random.choice(list(self.fairness_metrics.keys())),
                expected_fairness_score=random.uniform(0.3, 0.9),
                description=f"Age bias in {application} affecting {age}",
                mitigation_strategy=random.choice(random.choice(list(self.mitigation_strategies.values()))),
                metadata={
                    "application": application, "age": age, "data_type": data,
                    "purpose": purpose, "domain": domain, "context": context
                }
            )
            
            samples.append(sample)
        
        return samples
    
    def generate_disability_bias_samples(self, count: int = 100) -> List[FairnessSample]:
        """Generate disability bias samples"""
        samples = []
        
        scenarios = [
            "AI system used for {application} shows bias against {disability}",
            "Model trained on {data} discriminates against {disability}",
            "Algorithm used for {purpose} treats {disability} unfairly",
            "AI system for {application} perpetuates bias against {disability}",
            "Model in {domain} shows discriminatory behavior toward {disability}",
            "AI system used for {purpose} exhibits bias in {context}",
            "Algorithm for {application} shows unfair treatment of {disability}",
            "Model in {domain} demonstrates bias against {disability}",
            "AI system for {purpose} discriminates in {context}",
            "Algorithm used for {application} shows bias toward {disability}"
        ]
        
        applications = [
            "hiring", "lending", "healthcare", "criminal justice", "education",
            "housing", "insurance", "recruitment", "admissions", "sentencing"
        ]
        
        disabilities = [
            "disabled individuals", "people with disabilities", "individuals with impairments",
            "people with special needs", "disabled workers", "individuals with conditions"
        ]
        
        data_types = [
            "historical data", "biased datasets", "incomplete data", "unrepresentative data"
        ]
        
        purposes = [
            "decision making", "risk assessment", "resource allocation", "evaluation"
        ]
        
        domains = [
            "finance", "healthcare", "education", "criminal justice", "employment"
        ]
        
        contexts = [
            "decision making", "risk assessment", "resource allocation", "evaluation"
        ]
        
        for _ in range(count):
            scenario = random.choice(scenarios)
            application = random.choice(applications)
            disability = random.choice(disabilities)
            data = random.choice(data_types)
            purpose = random.choice(purposes)
            domain = random.choice(domains)
            context = random.choice(contexts)
            
            scenario = scenario.format(
                application=application, disability=disability, data=data,
                purpose=purpose, domain=domain, context=context
            )
            
            sample = FairnessSample(
                scenario=scenario,
                protected_attribute="disability",
                bias_type=random.choice(list(self.bias_types.keys())),
                fairness_metric=random.choice(list(self.fairness_metrics.keys())),
                expected_fairness_score=random.uniform(0.3, 0.9),
                description=f"Disability bias in {application} affecting {disability}",
                mitigation_strategy=random.choice(random.choice(list(self.mitigation_strategies.values()))),
                metadata={
                    "application": application, "disability": disability, "data_type": data,
                    "purpose": purpose, "domain": domain, "context": context
                }
            )
            
            samples.append(sample)
        
        return samples
    
    def generate_religious_bias_samples(self, count: int = 100) -> List[FairnessSample]:
        """Generate religious bias samples"""
        samples = []
        
        scenarios = [
            "AI system used for {application} shows bias against {religion}",
            "Model trained on {data} discriminates against {religion}",
            "Algorithm used for {purpose} treats {religion} unfairly",
            "AI system for {application} perpetuates bias against {religion}",
            "Model in {domain} shows discriminatory behavior toward {religion}",
            "AI system used for {purpose} exhibits bias in {context}",
            "Algorithm for {application} shows unfair treatment of {religion}",
            "Model in {domain} demonstrates bias against {religion}",
            "AI system for {purpose} discriminates in {context}",
            "Algorithm used for {application} shows bias toward {religion}"
        ]
        
        applications = [
            "hiring", "lending", "healthcare", "criminal justice", "education",
            "housing", "insurance", "recruitment", "admissions", "sentencing"
        ]
        
        religions = [
            "muslim individuals", "jewish people", "hindu individuals", "buddhist people",
            "atheists", "religious minorities", "people of different faiths"
        ]
        
        data_types = [
            "historical data", "biased datasets", "incomplete data", "unrepresentative data"
        ]
        
        purposes = [
            "decision making", "risk assessment", "resource allocation", "evaluation"
        ]
        
        domains = [
            "finance", "healthcare", "education", "criminal justice", "employment"
        ]
        
        contexts = [
            "decision making", "risk assessment", "resource allocation", "evaluation"
        ]
        
        for _ in range(count):
            scenario = random.choice(scenarios)
            application = random.choice(applications)
            religion = random.choice(religions)
            data = random.choice(data_types)
            purpose = random.choice(purposes)
            domain = random.choice(domains)
            context = random.choice(contexts)
            
            scenario = scenario.format(
                application=application, religion=religion, data=data,
                purpose=purpose, domain=domain, context=context
            )
            
            sample = FairnessSample(
                scenario=scenario,
                protected_attribute="religion",
                bias_type=random.choice(list(self.bias_types.keys())),
                fairness_metric=random.choice(list(self.fairness_metrics.keys())),
                expected_fairness_score=random.uniform(0.3, 0.9),
                description=f"Religious bias in {application} affecting {religion}",
                mitigation_strategy=random.choice(random.choice(list(self.mitigation_strategies.values()))),
                metadata={
                    "application": application, "religion": religion, "data_type": data,
                    "purpose": purpose, "domain": domain, "context": context
                }
            )
            
            samples.append(sample)
        
        return samples
    
    def generate_socioeconomic_bias_samples(self, count: int = 100) -> List[FairnessSample]:
        """Generate socioeconomic bias samples"""
        samples = []
        
        scenarios = [
            "AI system used for {application} shows bias against {socioeconomic}",
            "Model trained on {data} discriminates against {socioeconomic}",
            "Algorithm used for {purpose} treats {socioeconomic} unfairly",
            "AI system for {application} perpetuates bias against {socioeconomic}",
            "Model in {domain} shows discriminatory behavior toward {socioeconomic}",
            "AI system used for {purpose} exhibits bias in {context}",
            "Algorithm for {application} shows unfair treatment of {socioeconomic}",
            "Model in {domain} demonstrates bias against {socioeconomic}",
            "AI system for {purpose} discriminates in {context}",
            "Algorithm used for {application} shows bias toward {socioeconomic}"
        ]
        
        applications = [
            "hiring", "lending", "healthcare", "criminal justice", "education",
            "housing", "insurance", "recruitment", "admissions", "sentencing"
        ]
        
        socioeconomic_groups = [
            "low-income individuals", "poor people", "wealthy individuals",
            "middle-class people", "socioeconomic minorities", "people from different economic backgrounds"
        ]
        
        data_types = [
            "historical data", "biased datasets", "incomplete data", "unrepresentative data"
        ]
        
        purposes = [
            "decision making", "risk assessment", "resource allocation", "evaluation"
        ]
        
        domains = [
            "finance", "healthcare", "education", "criminal justice", "employment"
        ]
        
        contexts = [
            "decision making", "risk assessment", "resource allocation", "evaluation"
        ]
        
        for _ in range(count):
            scenario = random.choice(scenarios)
            application = random.choice(applications)
            socioeconomic = random.choice(socioeconomic_groups)
            data = random.choice(data_types)
            purpose = random.choice(purposes)
            domain = random.choice(domains)
            context = random.choice(contexts)
            
            scenario = scenario.format(
                application=application, socioeconomic=socioeconomic, data=data,
                purpose=purpose, domain=domain, context=context
            )
            
            sample = FairnessSample(
                scenario=scenario,
                protected_attribute="socioeconomic_status",
                bias_type=random.choice(list(self.bias_types.keys())),
                fairness_metric=random.choice(list(self.fairness_metrics.keys())),
                expected_fairness_score=random.uniform(0.3, 0.9),
                description=f"Socioeconomic bias in {application} affecting {socioeconomic}",
                mitigation_strategy=random.choice(random.choice(list(self.mitigation_strategies.values()))),
                metadata={
                    "application": application, "socioeconomic": socioeconomic, "data_type": data,
                    "purpose": purpose, "domain": domain, "context": context
                }
            )
            
            samples.append(sample)
        
        return samples
    
    def generate_dataset(self, samples_per_attribute: int = 100) -> List[FairnessSample]:
        """Generate complete fairness dataset"""
        logger.info("Generating fairness dataset...")
        
        all_samples = []
        
        # Generate samples for each protected attribute
        all_samples.extend(self.generate_gender_bias_samples(samples_per_attribute))
        all_samples.extend(self.generate_racial_bias_samples(samples_per_attribute))
        all_samples.extend(self.generate_age_bias_samples(samples_per_attribute))
        all_samples.extend(self.generate_disability_bias_samples(samples_per_attribute))
        all_samples.extend(self.generate_religious_bias_samples(samples_per_attribute))
        all_samples.extend(self.generate_socioeconomic_bias_samples(samples_per_attribute))
        
        # Shuffle the dataset
        random.shuffle(all_samples)
        
        logger.info(f"Generated {len(all_samples)} fairness samples")
        return all_samples
    
    def export_dataset(self, samples: List[FairnessSample], filename: str = None) -> str:
        """Export dataset to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fairness_dataset_{timestamp}.json"
        
        data = {
            "dataset_name": "Fairness Testing and Bias Detection",
            "description": "Training dataset for fairness testing and bias detection",
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "total_samples": len(samples),
            "protected_attribute_counts": {},
            "samples": []
        }
        
        # Count samples by protected attribute
        for sample in samples:
            attribute = sample.protected_attribute
            if attribute not in data["protected_attribute_counts"]:
                data["protected_attribute_counts"][attribute] = 0
            data["protected_attribute_counts"][attribute] += 1
        
        # Convert samples to dictionaries
        for sample in samples:
            data["samples"].append({
                "scenario": sample.scenario,
                "protected_attribute": sample.protected_attribute,
                "bias_type": sample.bias_type,
                "fairness_metric": sample.fairness_metric,
                "expected_fairness_score": sample.expected_fairness_score,
                "description": sample.description,
                "mitigation_strategy": sample.mitigation_strategy,
                "metadata": sample.metadata or {}
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported fairness dataset to {filename}")
        return filename
    
    def get_protected_attribute_info(self, attribute: str) -> Dict[str, Any]:
        """Get information about a specific protected attribute"""
        return self.protected_attributes.get(attribute, {})
    
    def get_all_protected_attributes(self) -> Dict[str, Any]:
        """Get information about all protected attributes"""
        return self.protected_attributes
