"""
Privacy Dataset Generator
Generates training datasets for privacy testing and data protection
"""

import json
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class PrivacySample:
    """Privacy sample data structure"""
    scenario: str
    privacy_risk: str
    data_type: str
    privacy_metric: str
    expected_privacy_score: float
    description: str
    protection_strategy: str
    metadata: Dict[str, Any] = None

class PrivacyDatasetGenerator:
    """Generates privacy testing and data protection datasets"""
    
    def __init__(self):
        self.privacy_risks = {
            "data_breach": {
                "description": "Unauthorized access to or disclosure of personal data",
                "examples": [
                    "Personal information exposed in data breach",
                    "Sensitive data accessed by unauthorized users",
                    "Private information leaked to third parties",
                    "Confidential data compromised by attackers",
                    "Personal details stolen by hackers"
                ]
            },
            "data_misuse": {
                "description": "Inappropriate use of personal data beyond intended purpose",
                "examples": [
                    "Personal data used for unauthorized purposes",
                    "Sensitive information shared without consent",
                    "Private data sold to third parties",
                    "Personal details used for targeted advertising",
                    "Confidential information used for profiling"
                ]
            },
            "surveillance": {
                "description": "Excessive monitoring and tracking of individuals",
                "examples": [
                    "Individuals tracked without consent",
                    "Personal behavior monitored continuously",
                    "Private activities recorded and stored",
                    "Personal data collected without knowledge",
                    "Individuals surveilled without authorization"
                ]
            },
            "profiling": {
                "description": "Creation of detailed profiles of individuals",
                "examples": [
                    "Detailed profiles created without consent",
                    "Personal characteristics analyzed and stored",
                    "Individual behavior patterns tracked",
                    "Personal preferences recorded and used",
                    "Individual characteristics profiled for targeting"
                ]
            },
            "discrimination": {
                "description": "Use of personal data to discriminate against individuals",
                "examples": [
                    "Personal data used to deny services",
                    "Sensitive information used for discrimination",
                    "Private data used to exclude individuals",
                    "Personal details used to treat individuals unfairly",
                    "Confidential information used to harm individuals"
                ]
            }
        }
        
        self.data_types = {
            "personal_identifiers": {
                "description": "Information that directly identifies individuals",
                "examples": ["names", "addresses", "phone numbers", "email addresses", "social security numbers"],
                "sensitivity": "high"
            },
            "biometric_data": {
                "description": "Biological characteristics used for identification",
                "examples": ["fingerprints", "facial recognition", "voice patterns", "iris scans", "DNA"],
                "sensitivity": "very_high"
            },
            "financial_data": {
                "description": "Information related to financial status and transactions",
                "examples": ["bank accounts", "credit scores", "income", "spending patterns", "investment portfolios"],
                "sensitivity": "high"
            },
            "health_data": {
                "description": "Information related to health and medical conditions",
                "examples": ["medical records", "diagnoses", "treatments", "medications", "health insurance"],
                "sensitivity": "very_high"
            },
            "location_data": {
                "description": "Information about where individuals are or have been",
                "examples": ["GPS coordinates", "addresses", "travel patterns", "frequent locations", "movement history"],
                "sensitivity": "medium"
            },
            "behavioral_data": {
                "description": "Information about how individuals behave and interact",
                "examples": ["browsing history", "purchase behavior", "social media activity", "communication patterns", "preferences"],
                "sensitivity": "medium"
            },
            "demographic_data": {
                "description": "Information about personal characteristics and background",
                "examples": ["age", "gender", "race", "religion", "education", "occupation"],
                "sensitivity": "low"
            }
        }
        
        self.privacy_metrics = {
            "differential_privacy": {
                "description": "Mathematical framework for privacy-preserving data analysis",
                "range": [0, 1],
                "ideal_value": 1
            },
            "k_anonymity": {
                "description": "Ensuring each record is indistinguishable from k-1 others",
                "range": [1, float('inf')],
                "ideal_value": 5
            },
            "l_diversity": {
                "description": "Ensuring sensitive attributes have sufficient diversity",
                "range": [1, float('inf')],
                "ideal_value": 3
            },
            "t_closeness": {
                "description": "Ensuring distribution of sensitive attributes is close to overall distribution",
                "range": [0, 1],
                "ideal_value": 0.1
            },
            "privacy_budget": {
                "description": "Amount of privacy budget consumed by queries",
                "range": [0, 1],
                "ideal_value": 0.1
            }
        }
        
        self.protection_strategies = {
            "data_minimization": [
                "Collect only necessary data",
                "Retain data only as long as needed",
                "Use data only for intended purposes",
                "Limit data collection to minimum required",
                "Regularly review and delete unnecessary data"
            ],
            "anonymization": [
                "Remove or mask identifying information",
                "Use pseudonymization techniques",
                "Apply generalization and suppression",
                "Use synthetic data generation",
                "Implement k-anonymity techniques"
            ],
            "encryption": [
                "Encrypt data at rest",
                "Encrypt data in transit",
                "Use strong encryption algorithms",
                "Implement key management systems",
                "Use homomorphic encryption"
            ],
            "access_controls": [
                "Implement role-based access control",
                "Use multi-factor authentication",
                "Implement least privilege principle",
                "Regular access reviews and audits",
                "Use data loss prevention tools"
            ],
            "privacy_by_design": [
                "Integrate privacy into system design",
                "Use privacy-preserving technologies",
                "Implement data protection by default",
                "Conduct privacy impact assessments",
                "Use privacy-enhancing technologies"
            ]
        }
    
    def generate_data_breach_samples(self, count: int = 100) -> List[PrivacySample]:
        """Generate data breach samples"""
        samples = []
        
        scenarios = [
            "Personal {data} exposed in {breach_type} breach",
            "Sensitive {data} accessed by {attacker}",
            "Private {data} leaked to {recipient}",
            "Confidential {data} compromised by {method}",
            "Personal {data} stolen by {attacker}",
            "Sensitive {data} exposed through {vulnerability}",
            "Private {data} accessed without {authorization}",
            "Confidential {data} disclosed to {recipient}",
            "Personal {data} compromised in {incident}",
            "Sensitive {data} leaked through {method}"
        ]
        
        data_types = [
            "information", "data", "details", "records", "files"
        ]
        
        breach_types = [
            "cyber", "security", "data", "privacy", "information"
        ]
        
        attackers = [
            "hackers", "cybercriminals", "unauthorized users", "malicious actors", "threat actors"
        ]
        
        recipients = [
            "third parties", "unauthorized users", "malicious actors", "cybercriminals", "hackers"
        ]
        
        methods = [
            "cyber attacks", "social engineering", "insider threats", "system vulnerabilities", "malware"
        ]
        
        vulnerabilities = [
            "system vulnerabilities", "security flaws", "configuration errors", "software bugs", "weak passwords"
        ]
        
        authorizations = [
            "authorization", "permission", "consent", "proper access", "legal basis"
        ]
        
        incidents = [
            "security incident", "cyber attack", "data breach", "privacy violation", "security breach"
        ]
        
        for _ in range(count):
            scenario = random.choice(scenarios)
            data_type = random.choice(data_types)
            breach_type = random.choice(breach_types)
            attacker = random.choice(attackers)
            recipient = random.choice(recipients)
            method = random.choice(methods)
            vulnerability = random.choice(vulnerabilities)
            authorization = random.choice(authorizations)
            incident = random.choice(incidents)
            
            scenario = scenario.format(
                data=data_type, breach_type=breach_type, attacker=attacker,
                recipient=recipient, method=method, vulnerability=vulnerability,
                authorization=authorization, incident=incident
            )
            
            sample = PrivacySample(
                scenario=scenario,
                privacy_risk="data_breach",
                data_type=random.choice(list(self.data_types.keys())),
                privacy_metric=random.choice(list(self.privacy_metrics.keys())),
                expected_privacy_score=random.uniform(0.1, 0.5),
                description=f"Data breach involving {data_type}",
                protection_strategy=random.choice(random.choice(list(self.protection_strategies.values()))),
                metadata={
                    "data_type": data_type, "breach_type": breach_type, "attacker": attacker,
                    "recipient": recipient, "method": method, "vulnerability": vulnerability
                }
            )
            
            samples.append(sample)
        
        return samples
    
    def generate_data_misuse_samples(self, count: int = 100) -> List[PrivacySample]:
        """Generate data misuse samples"""
        samples = []
        
        scenarios = [
            "Personal {data} used for {unauthorized_purpose}",
            "Sensitive {data} shared with {recipient} without {consent}",
            "Private {data} sold to {buyer} for {purpose}",
            "Personal {data} used for {activity} without {permission}",
            "Confidential {data} shared with {recipient} for {purpose}",
            "Personal {data} used for {activity} beyond {scope}",
            "Sensitive {data} shared with {recipient} for {unauthorized_purpose}",
            "Private {data} used for {activity} without {authorization}",
            "Personal {data} shared with {recipient} for {purpose}",
            "Confidential {data} used for {activity} without {consent}"
        ]
        
        data_types = [
            "information", "data", "details", "records", "files"
        ]
        
        unauthorized_purposes = [
            "unauthorized purposes", "commercial purposes", "marketing", "profiling", "targeting"
        ]
        
        recipients = [
            "third parties", "advertisers", "marketers", "data brokers", "commercial entities"
        ]
        
        consent_types = [
            "consent", "permission", "authorization", "proper consent", "informed consent"
        ]
        
        buyers = [
            "data brokers", "advertisers", "marketers", "third parties", "commercial entities"
        ]
        
        purposes = [
            "marketing", "advertising", "profiling", "targeting", "commercial purposes"
        ]
        
        activities = [
            "targeted advertising", "profiling", "marketing", "surveillance", "monitoring"
        ]
        
        permissions = [
            "permission", "consent", "authorization", "proper permission", "informed consent"
        ]
        
        scopes = [
            "intended scope", "authorized scope", "permitted scope", "legal scope", "agreed scope"
        ]
        
        authorizations = [
            "authorization", "permission", "consent", "proper authorization", "informed consent"
        ]
        
        for _ in range(count):
            scenario = random.choice(scenarios)
            data_type = random.choice(data_types)
            unauthorized_purpose = random.choice(unauthorized_purposes)
            recipient = random.choice(recipients)
            consent = random.choice(consent_types)
            buyer = random.choice(buyers)
            purpose = random.choice(purposes)
            activity = random.choice(activities)
            permission = random.choice(permissions)
            scope = random.choice(scopes)
            authorization = random.choice(authorizations)
            
            scenario = scenario.format(
                data=data_type, unauthorized_purpose=unauthorized_purpose, recipient=recipient,
                consent=consent, buyer=buyer, purpose=purpose, activity=activity,
                permission=permission, scope=scope, authorization=authorization
            )
            
            sample = PrivacySample(
                scenario=scenario,
                privacy_risk="data_misuse",
                data_type=random.choice(list(self.data_types.keys())),
                privacy_metric=random.choice(list(self.privacy_metrics.keys())),
                expected_privacy_score=random.uniform(0.2, 0.6),
                description=f"Data misuse involving {data_type}",
                protection_strategy=random.choice(random.choice(list(self.protection_strategies.values()))),
                metadata={
                    "data_type": data_type, "unauthorized_purpose": unauthorized_purpose,
                    "recipient": recipient, "consent": consent, "buyer": buyer
                }
            )
            
            samples.append(sample)
        
        return samples
    
    def generate_surveillance_samples(self, count: int = 100) -> List[PrivacySample]:
        """Generate surveillance samples"""
        samples = []
        
        scenarios = [
            "Individuals {tracked} without {consent}",
            "Personal {behavior} monitored {continuously}",
            "Private {activities} recorded and {stored}",
            "Personal {data} collected without {knowledge}",
            "Individuals {surveilled} without {authorization}",
            "Personal {behavior} tracked {continuously}",
            "Private {activities} monitored without {consent}",
            "Personal {data} collected {secretly}",
            "Individuals {monitored} without {permission}",
            "Personal {behavior} surveilled {continuously}"
        ]
        
        tracking_methods = [
            "tracked", "monitored", "surveilled", "watched", "observed"
        ]
        
        consent_types = [
            "consent", "permission", "authorization", "proper consent", "informed consent"
        ]
        
        behaviors = [
            "behavior", "activities", "movements", "interactions", "communications"
        ]
        
        continuous_terms = [
            "continuously", "constantly", "persistently", "regularly", "frequently"
        ]
        
        activities = [
            "activities", "behavior", "movements", "interactions", "communications"
        ]
        
        storage_methods = [
            "stored", "recorded", "saved", "archived", "retained"
        ]
        
        data_types = [
            "data", "information", "details", "records", "files"
        ]
        
        knowledge_types = [
            "knowledge", "awareness", "notification", "information", "disclosure"
        ]
        
        authorizations = [
            "authorization", "permission", "consent", "proper authorization", "informed consent"
        ]
        
        secret_terms = [
            "secretly", "covertly", "without notice", "without disclosure", "without knowledge"
        ]
        
        permissions = [
            "permission", "consent", "authorization", "proper permission", "informed consent"
        ]
        
        for _ in range(count):
            scenario = random.choice(scenarios)
            tracked = random.choice(tracking_methods)
            consent = random.choice(consent_types)
            behavior = random.choice(behaviors)
            continuously = random.choice(continuous_terms)
            activity = random.choice(activities)
            stored = random.choice(storage_methods)
            data = random.choice(data_types)
            knowledge = random.choice(knowledge_types)
            authorization = random.choice(authorizations)
            secretly = random.choice(secret_terms)
            permission = random.choice(permissions)
            
            scenario = scenario.format(
                tracked=tracked, consent=consent, behavior=behavior, continuously=continuously,
                activities=activity, stored=stored, data=data, knowledge=knowledge,
                authorization=authorization, surveilled=tracked, secretly=secretly,
                monitored=tracked, permission=permission
            )
            
            sample = PrivacySample(
                scenario=scenario,
                privacy_risk="surveillance",
                data_type=random.choice(list(self.data_types.keys())),
                privacy_metric=random.choice(list(self.privacy_metrics.keys())),
                expected_privacy_score=random.uniform(0.1, 0.4),
                description=f"Surveillance involving {behavior}",
                protection_strategy=random.choice(random.choice(list(self.protection_strategies.values()))),
                metadata={
                    "tracked": tracked, "consent": consent, "behavior": behavior,
                    "continuously": continuously, "activity": activity, "stored": stored
                }
            )
            
            samples.append(sample)
        
        return samples
    
    def generate_profiling_samples(self, count: int = 100) -> List[PrivacySample]:
        """Generate profiling samples"""
        samples = []
        
        scenarios = [
            "Detailed {profiles} created without {consent}",
            "Personal {characteristics} analyzed and {stored}",
            "Individual {behavior} patterns {tracked}",
            "Personal {preferences} recorded and {used}",
            "Individual {characteristics} profiled for {purpose}",
            "Detailed {profiles} built without {permission}",
            "Personal {characteristics} analyzed for {purpose}",
            "Individual {behavior} patterns {monitored}",
            "Personal {preferences} tracked and {stored}",
            "Individual {characteristics} profiled without {authorization}"
        ]
        
        profile_types = [
            "profiles", "dossiers", "records", "files", "dossiers"
        ]
        
        consent_types = [
            "consent", "permission", "authorization", "proper consent", "informed consent"
        ]
        
        characteristics = [
            "characteristics", "traits", "attributes", "features", "properties"
        ]
        
        storage_methods = [
            "stored", "recorded", "saved", "archived", "retained"
        ]
        
        behaviors = [
            "behavior", "activities", "patterns", "habits", "preferences"
        ]
        
        tracking_methods = [
            "tracked", "monitored", "recorded", "analyzed", "studied"
        ]
        
        preferences = [
            "preferences", "likes", "dislikes", "interests", "habits"
        ]
        
        usage_methods = [
            "used", "utilized", "exploited", "leveraged", "applied"
        ]
        
        purposes = [
            "targeting", "marketing", "advertising", "profiling", "surveillance"
        ]
        
        permissions = [
            "permission", "consent", "authorization", "proper permission", "informed consent"
        ]
        
        authorizations = [
            "authorization", "permission", "consent", "proper authorization", "informed consent"
        ]
        
        for _ in range(count):
            scenario = random.choice(scenarios)
            profile = random.choice(profile_types)
            consent = random.choice(consent_types)
            characteristic = random.choice(characteristics)
            stored = random.choice(storage_methods)
            behavior = random.choice(behaviors)
            tracked = random.choice(tracking_methods)
            preference = random.choice(preferences)
            used = random.choice(usage_methods)
            purpose = random.choice(purposes)
            permission = random.choice(permissions)
            authorization = random.choice(authorizations)
            
            scenario = scenario.format(
                profiles=profile, consent=consent, characteristics=characteristic,
                stored=stored, behavior=behavior, tracked=tracked, preferences=preference,
                used=used, purpose=purpose, permission=permission, authorization=authorization
            )
            
            sample = PrivacySample(
                scenario=scenario,
                privacy_risk="profiling",
                data_type=random.choice(list(self.data_types.keys())),
                privacy_metric=random.choice(list(self.privacy_metrics.keys())),
                expected_privacy_score=random.uniform(0.2, 0.7),
                description=f"Profiling involving {characteristic}",
                protection_strategy=random.choice(random.choice(list(self.protection_strategies.values()))),
                metadata={
                    "profile": profile, "consent": consent, "characteristic": characteristic,
                    "stored": stored, "behavior": behavior, "tracked": tracked
                }
            )
            
            samples.append(sample)
        
        return samples
    
    def generate_discrimination_samples(self, count: int = 100) -> List[PrivacySample]:
        """Generate discrimination samples"""
        samples = []
        
        scenarios = [
            "Personal {data} used to {discriminate} against {group}",
            "Sensitive {information} used for {discriminatory_purpose}",
            "Private {data} used to {exclude} {individuals}",
            "Personal {information} used to {treat} individuals {unfairly}",
            "Confidential {data} used to {harm} {individuals}",
            "Personal {data} used to {deny} {services} to {group}",
            "Sensitive {information} used for {discriminatory_activity}",
            "Private {data} used to {exclude} {group} from {opportunity}",
            "Personal {information} used to {treat} {group} {unfairly}",
            "Confidential {data} used to {discriminate} against {individuals}"
        ]
        
        data_types = [
            "data", "information", "details", "records", "files"
        ]
        
        discrimination_actions = [
            "discriminate", "exclude", "deny", "reject", "refuse"
        ]
        
        groups = [
            "certain groups", "specific individuals", "particular demographics", "targeted populations", "vulnerable groups"
        ]
        
        discriminatory_purposes = [
            "discriminatory purposes", "unfair treatment", "exclusion", "denial of services", "harmful targeting"
        ]
        
        individuals = [
            "individuals", "people", "users", "customers", "applicants"
        ]
        
        unfair_treatments = [
            "treat", "handle", "process", "evaluate", "assess"
        ]
        
        unfair_terms = [
            "unfairly", "discriminatorily", "unequally", "differently", "unjustly"
        ]
        
        harm_actions = [
            "harm", "hurt", "damage", "injure", "disadvantage"
        ]
        
        services = [
            "services", "opportunities", "benefits", "access", "resources"
        ]
        
        discriminatory_activities = [
            "discriminatory activities", "unfair practices", "exclusionary behavior", "biased treatment", "prejudicial actions"
        ]
        
        opportunities = [
            "opportunities", "services", "benefits", "access", "resources"
        ]
        
        for _ in range(count):
            scenario = random.choice(scenarios)
            data = random.choice(data_types)
            discriminate = random.choice(discrimination_actions)
            group = random.choice(groups)
            discriminatory_purpose = random.choice(discriminatory_purposes)
            individual = random.choice(individuals)
            unfair_treatment = random.choice(unfair_treatments)
            unfair = random.choice(unfair_terms)
            harm = random.choice(harm_actions)
            service = random.choice(services)
            discriminatory_activity = random.choice(discriminatory_activities)
            opportunity = random.choice(opportunities)
            
            scenario = scenario.format(
                data=data, discriminate=discriminate, group=group, discriminatory_purpose=discriminatory_purpose,
                individuals=individual, exclude=discriminate, treat=unfair_treatment, unfairly=unfair,
                harm=harm, deny=discriminate, services=service, discriminatory_activity=discriminatory_activity,
                opportunity=opportunity, information=data
            )
            
            sample = PrivacySample(
                scenario=scenario,
                privacy_risk="discrimination",
                data_type=random.choice(list(self.data_types.keys())),
                privacy_metric=random.choice(list(self.privacy_metrics.keys())),
                expected_privacy_score=random.uniform(0.1, 0.5),
                description=f"Discrimination using {data}",
                protection_strategy=random.choice(random.choice(list(self.protection_strategies.values()))),
                metadata={
                    "data": data, "discriminate": discriminate, "group": group,
                    "discriminatory_purpose": discriminatory_purpose, "individual": individual
                }
            )
            
            samples.append(sample)
        
        return samples
    
    def generate_dataset(self, samples_per_risk: int = 100) -> List[PrivacySample]:
        """Generate complete privacy dataset"""
        logger.info("Generating privacy dataset...")
        
        all_samples = []
        
        # Generate samples for each privacy risk
        all_samples.extend(self.generate_data_breach_samples(samples_per_risk))
        all_samples.extend(self.generate_data_misuse_samples(samples_per_risk))
        all_samples.extend(self.generate_surveillance_samples(samples_per_risk))
        all_samples.extend(self.generate_profiling_samples(samples_per_risk))
        all_samples.extend(self.generate_discrimination_samples(samples_per_risk))
        
        # Shuffle the dataset
        random.shuffle(all_samples)
        
        logger.info(f"Generated {len(all_samples)} privacy samples")
        return all_samples
    
    def export_dataset(self, samples: List[PrivacySample], filename: str = None) -> str:
        """Export dataset to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"privacy_dataset_{timestamp}.json"
        
        data = {
            "dataset_name": "Privacy Testing and Data Protection",
            "description": "Training dataset for privacy testing and data protection",
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "total_samples": len(samples),
            "privacy_risk_counts": {},
            "samples": []
        }
        
        # Count samples by privacy risk
        for sample in samples:
            risk = sample.privacy_risk
            if risk not in data["privacy_risk_counts"]:
                data["privacy_risk_counts"][risk] = 0
            data["privacy_risk_counts"][risk] += 1
        
        # Convert samples to dictionaries
        for sample in samples:
            data["samples"].append({
                "scenario": sample.scenario,
                "privacy_risk": sample.privacy_risk,
                "data_type": sample.data_type,
                "privacy_metric": sample.privacy_metric,
                "expected_privacy_score": sample.expected_privacy_score,
                "description": sample.description,
                "protection_strategy": sample.protection_strategy,
                "metadata": sample.metadata or {}
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported privacy dataset to {filename}")
        return filename
    
    def get_privacy_risk_info(self, risk: str) -> Dict[str, Any]:
        """Get information about a specific privacy risk"""
        return self.privacy_risks.get(risk, {})
    
    def get_all_privacy_risks(self) -> Dict[str, Any]:
        """Get information about all privacy risks"""
        return self.privacy_risks
