"""
NIST AI RMF Dataset Generator
Generates training datasets based on NIST AI Risk Management Framework
"""

import json
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class NISTSample:
    """NIST sample data structure"""
    scenario: str
    risk_level: str
    risk_category: str
    trustworthiness_dimension: str
    governance_requirement: str
    mitigation_strategy: str
    metadata: Dict[str, Any] = None

class NISTDatasetGenerator:
    """Generates NIST AI RMF training datasets"""
    
    def __init__(self):
        self.risk_categories = {
            "bias": {
                "description": "Unfair or discriminatory treatment of individuals or groups",
                "examples": [
                    "AI system discriminates against certain demographic groups",
                    "Model shows bias in hiring decisions",
                    "Algorithm treats different groups unfairly",
                    "AI system perpetuates existing societal biases",
                    "Model makes decisions based on protected characteristics"
                ]
            },
            "privacy": {
                "description": "Unauthorized access to or use of personal information",
                "examples": [
                    "AI system collects personal data without consent",
                    "Model processes sensitive information inappropriately",
                    "Algorithm shares personal data with third parties",
                    "AI system stores personal information insecurely",
                    "Model uses personal data for unintended purposes"
                ]
            },
            "robustness": {
                "description": "AI system's ability to perform reliably under various conditions",
                "examples": [
                    "AI system fails under adversarial conditions",
                    "Model performs poorly on edge cases",
                    "Algorithm is vulnerable to input manipulation",
                    "AI system fails in different environments",
                    "Model is sensitive to small input changes"
                ]
            },
            "explainability": {
                "description": "Ability to understand and explain AI system decisions",
                "examples": [
                    "AI system decisions cannot be explained",
                    "Model provides no reasoning for its outputs",
                    "Algorithm is a black box with no transparency",
                    "AI system decisions are not interpretable",
                    "Model lacks accountability mechanisms"
                ]
            },
            "safety": {
                "description": "Potential for AI system to cause physical or psychological harm",
                "examples": [
                    "AI system could cause physical injury",
                    "Model might harm users psychologically",
                    "Algorithm could lead to dangerous situations",
                    "AI system poses safety risks to operators",
                    "Model could cause environmental damage"
                ]
            }
        }
        
        self.trustworthiness_dimensions = {
            "accuracy": "The degree to which the AI system produces correct outputs",
            "fairness": "The degree to which the AI system treats all individuals and groups equitably",
            "robustness": "The degree to which the AI system performs reliably under various conditions",
            "explainability": "The degree to which the AI system's decisions can be understood and explained",
            "privacy": "The degree to which the AI system protects personal information"
        }
        
        self.governance_requirements = {
            "risk_management": "Establish and maintain AI risk management processes",
            "governance_framework": "Develop and implement AI governance framework",
            "oversight": "Establish AI oversight mechanisms and monitoring",
            "documentation": "Maintain comprehensive documentation of AI systems",
            "training": "Provide training on AI risks and mitigation strategies",
            "incident_response": "Develop and test incident response procedures",
            "compliance": "Ensure compliance with applicable laws and regulations",
            "audit": "Conduct regular audits of AI systems and processes"
        }
        
        self.mitigation_strategies = {
            "bias": [
                "Implement bias detection and mitigation techniques",
                "Use diverse and representative training data",
                "Regular bias testing and monitoring",
                "Human oversight and review processes",
                "Algorithmic auditing and fairness metrics"
            ],
            "privacy": [
                "Implement privacy-preserving techniques",
                "Data minimization and purpose limitation",
                "Strong access controls and encryption",
                "Regular privacy impact assessments",
                "User consent and data rights management"
            ],
            "robustness": [
                "Adversarial training and testing",
                "Input validation and sanitization",
                "Error handling and recovery mechanisms",
                "Regular security testing and updates",
                "Defense in depth strategies"
            ],
            "explainability": [
                "Implement explainable AI techniques",
                "Provide decision explanations and reasoning",
                "Human-in-the-loop processes",
                "Transparency and accountability measures",
                "Regular explainability assessments"
            ],
            "safety": [
                "Safety testing and validation",
                "Fail-safe mechanisms and controls",
                "Human oversight and intervention",
                "Regular safety assessments",
                "Emergency shutdown procedures"
            ]
        }
    
    def generate_bias_scenarios(self, count: int = 100) -> List[NISTSample]:
        """Generate bias-related scenarios"""
        samples = []
        
        scenarios = [
            "AI system used for {application} shows bias against {group}",
            "Model trained on {data} discriminates against {group}",
            "Algorithm used for {purpose} treats {group} unfairly",
            "AI system for {application} perpetuates bias against {group}",
            "Model in {domain} shows discriminatory behavior toward {group}",
            "AI system used for {purpose} exhibits bias in {context}",
            "Algorithm for {application} shows unfair treatment of {group}",
            "Model in {domain} demonstrates bias against {group}",
            "AI system for {purpose} discriminates in {context}",
            "Algorithm used for {application} shows bias toward {group}"
        ]
        
        applications = [
            "hiring", "lending", "healthcare", "criminal justice", "education",
            "housing", "insurance", "recruitment", "admissions", "sentencing"
        ]
        
        groups = [
            "women", "minorities", "elderly", "disabled", "LGBTQ+",
            "immigrants", "low-income", "certain ethnicities", "specific religions"
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
            group = random.choice(groups)
            data = random.choice(data_types)
            purpose = random.choice(purposes)
            domain = random.choice(domains)
            context = random.choice(contexts)
            
            scenario = scenario.format(
                application=application, group=group, data=data,
                purpose=purpose, domain=domain, context=context
            )
            
            sample = NISTSample(
                scenario=scenario,
                risk_level=random.choice(["Low", "Medium", "High"]),
                risk_category="bias",
                trustworthiness_dimension="fairness",
                governance_requirement=random.choice(list(self.governance_requirements.keys())),
                mitigation_strategy=random.choice(self.mitigation_strategies["bias"]),
                metadata={
                    "application": application, "group": group, "data_type": data,
                    "purpose": purpose, "domain": domain, "context": context
                }
            )
            
            samples.append(sample)
        
        return samples
    
    def generate_privacy_scenarios(self, count: int = 100) -> List[NISTSample]:
        """Generate privacy-related scenarios"""
        samples = []
        
        scenarios = [
            "AI system collects {data} without proper {consent}",
            "Model processes {sensitive} information in {manner}",
            "Algorithm shares {data} with {recipients} without authorization",
            "AI system stores {information} in {storage} manner",
            "Model uses {data} for {purpose} without consent",
            "AI system accesses {sensitive} data without {permission}",
            "Algorithm processes {information} beyond {scope}",
            "AI system retains {data} longer than {duration}",
            "Model shares {sensitive} information with {parties}",
            "AI system uses {data} for {unintended} purposes"
        ]
        
        data_types = [
            "personal information", "sensitive data", "private information", "confidential data"
        ]
        
        consent_types = [
            "informed consent", "explicit consent", "proper authorization", "user consent"
        ]
        
        sensitive_info = [
            "health information", "financial data", "biometric data", "location data"
        ]
        
        manners = [
            "inappropriate", "unauthorized", "unlawful", "improper"
        ]
        
        recipients = [
            "third parties", "external organizations", "unauthorized users", "commercial entities"
        ]
        
        storage_types = [
            "insecure", "unencrypted", "unprotected", "vulnerable"
        ]
        
        purposes = [
            "commercial", "research", "marketing", "surveillance"
        ]
        
        permissions = [
            "proper permission", "authorization", "consent", "legal basis"
        ]
        
        scopes = [
            "intended scope", "authorized scope", "permitted scope", "legal scope"
        ]
        
        durations = [
            "necessary", "authorized", "permitted", "legal"
        ]
        
        parties = [
            "unauthorized parties", "third parties", "external entities", "commercial partners"
        ]
        
        unintended_purposes = [
            "unintended", "unauthorized", "commercial", "surveillance"
        ]
        
        for _ in range(count):
            scenario = random.choice(scenarios)
            data = random.choice(data_types)
            consent = random.choice(consent_types)
            sensitive = random.choice(sensitive_info)
            manner = random.choice(manners)
            recipient = random.choice(recipients)
            storage = random.choice(storage_types)
            purpose = random.choice(purposes)
            permission = random.choice(permissions)
            scope = random.choice(scopes)
            duration = random.choice(durations)
            party = random.choice(parties)
            unintended = random.choice(unintended_purposes)
            
            scenario = scenario.format(
                data=data, consent=consent, sensitive=sensitive, manner=manner,
                recipients=recipient, information=sensitive, storage=storage,
                purpose=purpose, permission=permission, scope=scope,
                duration=duration, parties=party, unintended=unintended
            )
            
            sample = NISTSample(
                scenario=scenario,
                risk_level=random.choice(["Low", "Medium", "High"]),
                risk_category="privacy",
                trustworthiness_dimension="privacy",
                governance_requirement=random.choice(list(self.governance_requirements.keys())),
                mitigation_strategy=random.choice(self.mitigation_strategies["privacy"]),
                metadata={
                    "data_type": data, "consent": consent, "sensitive": sensitive,
                    "manner": manner, "recipient": recipient, "storage": storage
                }
            )
            
            samples.append(sample)
        
        return samples
    
    def generate_robustness_scenarios(self, count: int = 100) -> List[NISTSample]:
        """Generate robustness-related scenarios"""
        samples = []
        
        scenarios = [
            "AI system fails under {condition} conditions",
            "Model performs poorly on {case} cases",
            "Algorithm is vulnerable to {attack} attacks",
            "AI system fails in {environment} environment",
            "Model is sensitive to {change} changes",
            "AI system fails when {situation} occurs",
            "Model performs poorly under {stress} stress",
            "Algorithm fails when {input} input is provided",
            "AI system fails in {scenario} scenario",
            "Model is vulnerable to {manipulation} manipulation"
        ]
        
        conditions = [
            "adversarial", "noisy", "unexpected", "extreme", "hostile"
        ]
        
        cases = [
            "edge", "corner", "unusual", "rare", "extreme"
        ]
        
        attacks = [
            "adversarial", "poisoning", "evasion", "extraction", "inference"
        ]
        
        environments = [
            "production", "real-world", "different", "hostile", "unexpected"
        ]
        
        changes = [
            "small input", "minor parameter", "slight modification", "small perturbation"
        ]
        
        situations = [
            "input is corrupted", "data is missing", "system is overloaded", "network is slow"
        ]
        
        stresses = [
            "high load", "resource constraints", "time pressure", "computational"
        ]
        
        inputs = [
            "malicious", "corrupted", "unexpected", "adversarial", "poisoned"
        ]
        
        scenarios_list = [
            "real-world deployment", "adversarial testing", "stress testing", "edge case testing"
        ]
        
        manipulations = [
            "input", "parameter", "model", "data", "system"
        ]
        
        for _ in range(count):
            scenario = random.choice(scenarios)
            condition = random.choice(conditions)
            case = random.choice(cases)
            attack = random.choice(attacks)
            environment = random.choice(environments)
            change = random.choice(changes)
            situation = random.choice(situations)
            stress = random.choice(stresses)
            input_type = random.choice(inputs)
            scenario_type = random.choice(scenarios_list)
            manipulation = random.choice(manipulations)
            
            scenario = scenario.format(
                condition=condition, case=case, attack=attack, environment=environment,
                change=change, situation=situation, stress=stress, input=input_type,
                scenario=scenario_type, manipulation=manipulation
            )
            
            sample = NISTSample(
                scenario=scenario,
                risk_level=random.choice(["Low", "Medium", "High"]),
                risk_category="robustness",
                trustworthiness_dimension="robustness",
                governance_requirement=random.choice(list(self.governance_requirements.keys())),
                mitigation_strategy=random.choice(self.mitigation_strategies["robustness"]),
                metadata={
                    "condition": condition, "case": case, "attack": attack,
                    "environment": environment, "change": change, "situation": situation
                }
            )
            
            samples.append(sample)
        
        return samples
    
    def generate_explainability_scenarios(self, count: int = 100) -> List[NISTSample]:
        """Generate explainability-related scenarios"""
        samples = []
        
        scenarios = [
            "AI system decisions cannot be {explained}",
            "Model provides no {reasoning} for its outputs",
            "Algorithm is a {box} with no transparency",
            "AI system decisions are not {interpretable}",
            "Model lacks {mechanism} mechanisms",
            "AI system fails to provide {explanation}",
            "Model decisions are {opaque} and unclear",
            "Algorithm lacks {transparency} in decision making",
            "AI system provides {insufficient} explanations",
            "Model fails to justify its {decisions}"
        ]
        
        explanations = [
            "explained", "understood", "interpreted", "clarified"
        ]
        
        reasoning_types = [
            "reasoning", "explanation", "justification", "rationale"
        ]
        
        box_types = [
            "black box", "opaque system", "unexplainable model", "mysterious algorithm"
        ]
        
        interpretability = [
            "interpretable", "understandable", "explainable", "transparent"
        ]
        
        mechanisms = [
            "accountability", "transparency", "explainability", "interpretability"
        ]
        
        explanation_types = [
            "adequate explanation", "sufficient reasoning", "clear justification", "understandable rationale"
        ]
        
        opacity_levels = [
            "opaque", "unclear", "mysterious", "unexplainable"
        ]
        
        transparency_types = [
            "transparency", "clarity", "openness", "visibility"
        ]
        
        sufficiency = [
            "sufficient", "adequate", "clear", "comprehensive"
        ]
        
        decision_types = [
            "decisions", "outputs", "predictions", "recommendations"
        ]
        
        for _ in range(count):
            scenario = random.choice(scenarios)
            explained = random.choice(explanations)
            reasoning = random.choice(reasoning_types)
            box = random.choice(box_types)
            interpretable = random.choice(interpretability)
            mechanism = random.choice(mechanisms)
            explanation = random.choice(explanation_types)
            opaque = random.choice(opacity_levels)
            transparency = random.choice(transparency_types)
            sufficient = random.choice(sufficiency)
            decisions = random.choice(decision_types)
            
            scenario = scenario.format(
                explained=explained, reasoning=reasoning, box=box, interpretable=interpretable,
                mechanism=mechanism, explanation=explanation, opaque=opaque,
                transparency=transparency, insufficient=sufficient, decisions=decisions
            )
            
            sample = NISTSample(
                scenario=scenario,
                risk_level=random.choice(["Low", "Medium", "High"]),
                risk_category="explainability",
                trustworthiness_dimension="explainability",
                governance_requirement=random.choice(list(self.governance_requirements.keys())),
                mitigation_strategy=random.choice(self.mitigation_strategies["explainability"]),
                metadata={
                    "explained": explained, "reasoning": reasoning, "box": box,
                    "interpretable": interpretable, "mechanism": mechanism
                }
            )
            
            samples.append(sample)
        
        return samples
    
    def generate_safety_scenarios(self, count: int = 100) -> List[NISTSample]:
        """Generate safety-related scenarios"""
        samples = []
        
        scenarios = [
            "AI system could cause {harm} to {target}",
            "Model might {action} users in {context}",
            "Algorithm could lead to {situation}",
            "AI system poses {risk} to {stakeholders}",
            "Model could cause {damage} in {environment}",
            "AI system might {behavior} in {scenario}",
            "Algorithm could {outcome} when {condition}",
            "AI system poses {threat} to {victims}",
            "Model could {result} in {situation}",
            "AI system might {consequence} when {trigger}"
        ]
        
        harms = [
            "physical injury", "psychological harm", "financial loss", "reputational damage"
        ]
        
        targets = [
            "users", "operators", "patients", "customers", "employees"
        ]
        
        actions = [
            "harm", "injure", "endanger", "threaten", "damage"
        ]
        
        contexts = [
            "medical settings", "autonomous vehicles", "industrial environments", "home automation"
        ]
        
        situations = [
            "dangerous situations", "hazardous conditions", "unsafe environments", "risky scenarios"
        ]
        
        risks = [
            "safety risks", "health hazards", "security threats", "operational dangers"
        ]
        
        stakeholders = [
            "operators", "users", "patients", "customers", "employees"
        ]
        
        damages = [
            "environmental damage", "property damage", "data loss", "system failure"
        ]
        
        environments = [
            "production", "real-world", "critical", "safety-sensitive"
        ]
        
        behaviors = [
            "malfunction", "fail", "behave unpredictably", "act dangerously"
        ]
        
        scenarios_list = [
            "emergency situations", "high-stress conditions", "unexpected events", "edge cases"
        ]
        
        outcomes = [
            "cause accidents", "create hazards", "endanger lives", "damage property"
        ]
        
        conditions = [
            "malfunctioning", "under stress", "in unexpected situations", "when overloaded"
        ]
        
        threats = [
            "safety threats", "health hazards", "security risks", "operational dangers"
        ]
        
        victims = [
            "operators", "users", "patients", "customers", "employees"
        ]
        
        results = [
            "cause harm", "create danger", "endanger safety", "pose risks"
        ]
        
        consequences = [
            "cause injury", "create hazards", "endanger safety", "pose threats"
        ]
        
        triggers = [
            "malfunctioning", "under stress", "in unexpected situations", "when overloaded"
        ]
        
        for _ in range(count):
            scenario = random.choice(scenarios)
            harm = random.choice(harms)
            target = random.choice(targets)
            action = random.choice(actions)
            context = random.choice(contexts)
            situation = random.choice(situations)
            risk = random.choice(risks)
            stakeholder = random.choice(stakeholders)
            damage = random.choice(damages)
            environment = random.choice(environments)
            behavior = random.choice(behaviors)
            scenario_type = random.choice(scenarios_list)
            outcome = random.choice(outcomes)
            condition = random.choice(conditions)
            threat = random.choice(threats)
            victim = random.choice(victims)
            result = random.choice(results)
            consequence = random.choice(consequences)
            trigger = random.choice(triggers)
            
            scenario = scenario.format(
                harm=harm, target=target, action=action, context=context,
                situation=situation, risk=risk, stakeholders=stakeholder,
                damage=damage, environment=environment, behavior=behavior,
                scenario=scenario_type, outcome=outcome, condition=condition,
                threat=threat, victims=victim, result=result, consequence=consequence,
                trigger=trigger
            )
            
            sample = NISTSample(
                scenario=scenario,
                risk_level=random.choice(["Low", "Medium", "High"]),
                risk_category="safety",
                trustworthiness_dimension="safety",
                governance_requirement=random.choice(list(self.governance_requirements.keys())),
                mitigation_strategy=random.choice(self.mitigation_strategies["safety"]),
                metadata={
                    "harm": harm, "target": target, "action": action, "context": context,
                    "situation": situation, "risk": risk, "stakeholder": stakeholder
                }
            )
            
            samples.append(sample)
        
        return samples
    
    def generate_dataset(self, samples_per_category: int = 100) -> List[NISTSample]:
        """Generate complete NIST AI RMF dataset"""
        logger.info("Generating NIST AI RMF dataset...")
        
        all_samples = []
        
        # Generate samples for each risk category
        all_samples.extend(self.generate_bias_scenarios(samples_per_category))
        all_samples.extend(self.generate_privacy_scenarios(samples_per_category))
        all_samples.extend(self.generate_robustness_scenarios(samples_per_category))
        all_samples.extend(self.generate_explainability_scenarios(samples_per_category))
        all_samples.extend(self.generate_safety_scenarios(samples_per_category))
        
        # Shuffle the dataset
        random.shuffle(all_samples)
        
        logger.info(f"Generated {len(all_samples)} NIST samples")
        return all_samples
    
    def export_dataset(self, samples: List[NISTSample], filename: str = None) -> str:
        """Export dataset to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"nist_dataset_{timestamp}.json"
        
        data = {
            "dataset_name": "NIST AI RMF",
            "description": "Training dataset for NIST AI Risk Management Framework",
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "total_samples": len(samples),
            "risk_category_counts": {},
            "samples": []
        }
        
        # Count samples by risk category
        for sample in samples:
            category = sample.risk_category
            if category not in data["risk_category_counts"]:
                data["risk_category_counts"][category] = 0
            data["risk_category_counts"][category] += 1
        
        # Convert samples to dictionaries
        for sample in samples:
            data["samples"].append({
                "scenario": sample.scenario,
                "risk_level": sample.risk_level,
                "risk_category": sample.risk_category,
                "trustworthiness_dimension": sample.trustworthiness_dimension,
                "governance_requirement": sample.governance_requirement,
                "mitigation_strategy": sample.mitigation_strategy,
                "metadata": sample.metadata or {}
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported NIST dataset to {filename}")
        return filename
    
    def get_risk_category_info(self, category: str) -> Dict[str, Any]:
        """Get information about a specific risk category"""
        return self.risk_categories.get(category, {})
    
    def get_all_risk_categories(self) -> Dict[str, Any]:
        """Get information about all risk categories"""
        return self.risk_categories
