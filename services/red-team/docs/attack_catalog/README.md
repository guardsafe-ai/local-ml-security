# Attack Catalog

## Overview

This catalog provides comprehensive documentation of all attack methods available in the AI Red Team Security Service. Each attack is categorized by type, severity, and target model architecture.

## Attack Categories

### 1. Adversarial Attacks

#### Gradient-Based Attacks

##### FGSM (Fast Gradient Sign Method)
- **Description**: Single-step gradient-based attack
- **Target**: Neural networks with differentiable loss functions
- **Severity**: Medium
- **Parameters**:
  - `epsilon`: Perturbation magnitude (default: 0.1)
  - `targeted`: Whether to use targeted attack (default: False)
- **Example**:
  ```python
  from services.red_team.attacks import FGSMAttack
  
  attack = FGSMAttack(epsilon=0.1)
  result = attack.execute(model, data)
  ```

##### PGD (Projected Gradient Descent)
- **Description**: Multi-step gradient-based attack with projection
- **Target**: Neural networks with differentiable loss functions
- **Severity**: High
- **Parameters**:
  - `epsilon`: Perturbation magnitude (default: 0.1)
  - `step_size`: Step size for each iteration (default: 0.01)
  - `max_iterations`: Maximum number of iterations (default: 40)
- **Example**:
  ```python
  from services.red_team.attacks import PGDAttack
  
  attack = PGDAttack(epsilon=0.1, step_size=0.01, max_iterations=40)
  result = attack.execute(model, data)
  ```

##### C&W (Carlini & Wagner)
- **Description**: Optimization-based attack with L2, L0, and L∞ norms
- **Target**: Neural networks with differentiable loss functions
- **Severity**: High
- **Parameters**:
  - `norm`: Norm to use (L2, L0, L∞) (default: L2)
  - `confidence`: Confidence margin (default: 0)
  - `max_iterations`: Maximum iterations (default: 1000)
- **Example**:
  ```python
  from services.red_team.attacks import CWAttack
  
  attack = CWAttack(norm='L2', confidence=0, max_iterations=1000)
  result = attack.execute(model, data)
  ```

##### AutoAttack
- **Description**: Ensemble of parameter-free attacks
- **Target**: Neural networks with differentiable loss functions
- **Severity**: Critical
- **Parameters**:
  - `max_iterations`: Maximum iterations (default: 100)
  - `version`: AutoAttack version (default: 'standard')
- **Example**:
  ```python
  from services.red_team.attacks import AutoAttack
  
  attack = AutoAttack(max_iterations=100)
  result = attack.execute(model, data)
  ```

##### DeepFool
- **Description**: Minimal perturbation attack
- **Target**: Neural networks with differentiable loss functions
- **Severity**: Medium
- **Parameters**:
  - `max_iterations`: Maximum iterations (default: 50)
  - `overshoot`: Overshoot parameter (default: 0.02)
- **Example**:
  ```python
  from services.red_team.attacks import DeepFoolAttack
  
  attack = DeepFoolAttack(max_iterations=50, overshoot=0.02)
  result = attack.execute(model, data)
  ```

#### Word-Level Attacks

##### TextFooler
- **Description**: Word-level adversarial attack for text
- **Target**: Text classification models
- **Severity**: High
- **Parameters**:
  - `max_candidates`: Maximum candidate words (default: 50)
  - `threshold`: Similarity threshold (default: 0.5)
  - `use_semantic_similarity`: Use semantic similarity (default: True)
- **Example**:
  ```python
  from services.red_team.attacks import TextFoolerAttack
  
  attack = TextFoolerAttack(max_candidates=50, threshold=0.5)
  result = attack.execute(model, text_data)
  ```

##### BERT-Attack
- **Description**: BERT-based word substitution attack
- **Target**: Text classification models
- **Severity**: High
- **Parameters**:
  - `max_candidates`: Maximum candidate words (default: 50)
  - `threshold`: Similarity threshold (default: 0.5)
  - `use_semantic_similarity`: Use semantic similarity (default: True)
- **Example**:
  ```python
  from services.red_team.attacks import BERTAttack
  
  attack = BERTAttack(max_candidates=50, threshold=0.5)
  result = attack.execute(model, text_data)
  ```

##### HotFlip
- **Description**: Character-level adversarial attack
- **Target**: Text classification models
- **Severity**: Medium
- **Parameters**:
  - `max_changes`: Maximum character changes (default: 10)
  - `use_semantic_similarity`: Use semantic similarity (default: True)
- **Example**:
  ```python
  from services.red_team.attacks import HotFlipAttack
  
  attack = HotFlipAttack(max_changes=10)
  result = attack.execute(model, text_data)
  ```

#### Multi-Turn Attacks

##### TAP (Tree of Attacks with Pruning)
- **Description**: Systematic attack generation using tree search
- **Target**: Large language models
- **Severity**: Critical
- **Parameters**:
  - `max_depth`: Maximum tree depth (default: 5)
  - `beam_size`: Beam search size (default: 10)
  - `temperature`: Sampling temperature (default: 0.7)
- **Example**:
  ```python
  from services.red_team.attacks import TAPAttack
  
  attack = TAPAttack(max_depth=5, beam_size=10, temperature=0.7)
  result = attack.execute(model, prompt)
  ```

##### PAIR (Prompt Automatic Iterative Refinement)
- **Description**: Iterative prompt optimization
- **Target**: Large language models
- **Severity**: High
- **Parameters**:
  - `max_iterations`: Maximum iterations (default: 10)
  - `temperature`: Sampling temperature (default: 0.7)
  - `use_semantic_similarity`: Use semantic similarity (default: True)
- **Example**:
  ```python
  from services.red_team.attacks import PAIRAttack
  
  attack = PAIRAttack(max_iterations=10, temperature=0.7)
  result = attack.execute(model, prompt)
  ```

##### Crescendo
- **Description**: Escalating attack sequences
- **Target**: Large language models
- **Severity**: High
- **Parameters**:
  - `max_steps`: Maximum escalation steps (default: 5)
  - `escalation_factor`: Escalation factor (default: 1.5)
- **Example**:
  ```python
  from services.red_team.attacks import CrescendoAttack
  
  attack = CrescendoAttack(max_steps=5, escalation_factor=1.5)
  result = attack.execute(model, prompt)
  ```

#### Agent-Specific Attacks

##### Tool Injection
- **Description**: Malicious tool injection in agent systems
- **Target**: Agent-based systems with tool use
- **Severity**: Critical
- **Parameters**:
  - `injection_type`: Type of injection (default: "function")
  - `payload`: Injection payload (default: "malicious_function")
- **Example**:
  ```python
  from services.red_team.attacks import ToolInjectionAttack
  
  attack = ToolInjectionAttack(injection_type="function", payload="malicious_function")
  result = attack.execute(agent, input_data)
  ```

##### Prompt Leaking
- **Description**: System prompt extraction
- **Target**: Large language models
- **Severity**: High
- **Parameters**:
  - `extraction_method`: Extraction method (default: "direct")
  - `max_attempts`: Maximum attempts (default: 10)
- **Example**:
  ```python
  from services.red_team.attacks import PromptLeakingAttack
  
  attack = PromptLeakingAttack(extraction_method="direct", max_attempts=10)
  result = attack.execute(model, input_data)
  ```

##### Recursive Attacks
- **Description**: Self-replicating attacks
- **Target**: Large language models
- **Severity**: Critical
- **Parameters**:
  - `max_depth`: Maximum recursion depth (default: 5)
  - `replication_factor`: Replication factor (default: 2)
- **Example**:
  ```python
  from services.red_team.attacks import RecursiveAttack
  
  attack = RecursiveAttack(max_depth=5, replication_factor=2)
  result = attack.execute(model, input_data)
  ```

### 2. Traditional ML Attacks

#### Evasion Attacks
- **Description**: Input manipulation for misclassification
- **Target**: Traditional ML models (sklearn, XGBoost)
- **Severity**: Medium
- **Parameters**:
  - `perturbation_budget`: Maximum perturbation (default: 0.1)
  - `max_iterations`: Maximum iterations (default: 100)
- **Example**:
  ```python
  from services.red_team.attacks import EvasionAttack
  
  attack = EvasionAttack(perturbation_budget=0.1, max_iterations=100)
  result = attack.execute(model, data)
  ```

#### Poisoning Attacks
- **Description**: Training data corruption
- **Target**: Traditional ML models
- **Severity**: High
- **Parameters**:
  - `poison_ratio`: Ratio of poisoned data (default: 0.1)
  - `poison_type`: Type of poisoning (default: "label_flip")
- **Example**:
  ```python
  from services.red_team.attacks import PoisoningAttack
  
  attack = PoisoningAttack(poison_ratio=0.1, poison_type="label_flip")
  result = attack.execute(model, training_data)
  ```

#### Model Extraction Attacks
- **Description**: Model architecture and parameters theft
- **Target**: Traditional ML models
- **Severity**: High
- **Parameters**:
  - `extraction_method`: Extraction method (default: "query_based")
  - `max_queries`: Maximum queries (default: 1000)
- **Example**:
  ```python
  from services.red_team.attacks import ModelExtractionAttack
  
  attack = ModelExtractionAttack(extraction_method="query_based", max_queries=1000)
  result = attack.execute(model, data)
  ```

#### Membership Inference Attacks
- **Description**: Training data membership detection
- **Target**: Traditional ML models
- **Severity**: Medium
- **Parameters**:
  - `inference_method`: Inference method (default: "shadow_model")
  - `confidence_threshold`: Confidence threshold (default: 0.5)
- **Example**:
  ```python
  from services.red_team.attacks import MembershipInferenceAttack
  
  attack = MembershipInferenceAttack(inference_method="shadow_model", confidence_threshold=0.5)
  result = attack.execute(model, data)
  ```

### 3. Privacy Attacks

#### Membership Inference
- **Description**: Determine if data was in training set
- **Target**: All ML models
- **Severity**: Medium
- **Parameters**:
  - `inference_method`: Inference method (default: "shadow_model")
  - `confidence_threshold`: Confidence threshold (default: 0.5)
- **Example**:
  ```python
  from services.red_team.attacks import MembershipInferenceAttack
  
  attack = MembershipInferenceAttack(inference_method="shadow_model", confidence_threshold=0.5)
  result = attack.execute(model, data)
  ```

#### Model Inversion
- **Description**: Reconstruct training data from model
- **Target**: All ML models
- **Severity**: High
- **Parameters**:
  - `inversion_method`: Inversion method (default: "gradient_based")
  - `max_iterations`: Maximum iterations (default: 1000)
- **Example**:
  ```python
  from services.red_team.attacks import ModelInversionAttack
  
  attack = ModelInversionAttack(inversion_method="gradient_based", max_iterations=1000)
  result = attack.execute(model, data)
  ```

#### Data Extraction
- **Description**: Extract sensitive information from model
- **Target**: All ML models
- **Severity**: Critical
- **Parameters**:
  - `extraction_method`: Extraction method (default: "gradient_based")
  - `max_iterations`: Maximum iterations (default: 1000)
- **Example**:
  ```python
  from services.red_team.attacks import DataExtractionAttack
  
  attack = DataExtractionAttack(extraction_method="gradient_based", max_iterations=1000)
  result = attack.execute(model, data)
  ```

## Attack Execution

### Basic Execution

```python
from services.red_team import RedTeamService

# Initialize service
red_team = RedTeamService()

# Configure attack
attack_config = {
    "attack_type": "fgsm",
    "parameters": {
        "epsilon": 0.1,
        "targeted": False
    }
}

# Execute attack
result = red_team.execute_attack(
    model=model,
    data=data,
    attack_config=attack_config
)

# Analyze results
print(f"Attack success: {result['success']}")
print(f"Confidence: {result['confidence']}")
print(f"Perturbation: {result['perturbation']}")
```

### Batch Execution

```python
# Configure multiple attacks
attack_configs = [
    {
        "attack_type": "fgsm",
        "parameters": {"epsilon": 0.1}
    },
    {
        "attack_type": "pgd",
        "parameters": {"epsilon": 0.1, "max_iterations": 40}
    },
    {
        "attack_type": "c&w",
        "parameters": {"norm": "L2", "confidence": 0}
    }
]

# Execute batch
results = red_team.execute_attack_batch(
    model=model,
    data=data,
    attack_configs=attack_configs
)

# Analyze results
for i, result in enumerate(results):
    print(f"Attack {i+1}: {result['attack_type']} - Success: {result['success']}")
```

### Custom Attack Development

```python
from services.red_team.attacks import BaseAttack

class CustomAttack(BaseAttack):
    def __init__(self, config):
        super().__init__(config)
        self.name = "Custom Attack"
        self.description = "Custom attack implementation"
    
    def execute(self, model, data):
        # Implement attack logic
        perturbed_data = self.perturb_data(data)
        predictions = model.predict(perturbed_data)
        
        return {
            "success": self.evaluate_success(predictions),
            "confidence": self.calculate_confidence(predictions),
            "perturbation": self.calculate_perturbation(data, perturbed_data)
        }
    
    def perturb_data(self, data):
        # Custom perturbation logic
        pass
    
    def evaluate_success(self, predictions):
        # Success evaluation logic
        pass

# Register custom attack
red_team.register_attack(CustomAttack)
```

## Attack Results

### Result Structure

```python
{
    "attack_id": "unique_attack_id",
    "attack_type": "fgsm",
    "success": True,
    "confidence": 0.85,
    "perturbation": 0.05,
    "original_prediction": "class_1",
    "adversarial_prediction": "class_2",
    "execution_time": 1.23,
    "metadata": {
        "parameters": {"epsilon": 0.1},
        "model_info": {"type": "neural_network"},
        "data_info": {"shape": [100, 10]}
    }
}
```

### Result Analysis

```python
# Analyze attack results
analysis = red_team.analyze_results(results)

print(f"Total attacks: {analysis['total_attacks']}")
print(f"Successful attacks: {analysis['successful_attacks']}")
print(f"Success rate: {analysis['success_rate']:.2%}")
print(f"Average confidence: {analysis['average_confidence']:.2f}")
print(f"Average perturbation: {analysis['average_perturbation']:.2f}")

# Get top attacks by success rate
top_attacks = analysis['top_attacks']
for attack in top_attacks:
    print(f"{attack['attack_type']}: {attack['success_rate']:.2%}")
```

## Best Practices

### 1. Attack Selection
- Choose attacks appropriate for your model type
- Consider attack severity and impact
- Balance attack effectiveness with computational cost
- Use ensemble attacks for comprehensive testing

### 2. Parameter Tuning
- Start with default parameters
- Adjust parameters based on model characteristics
- Use grid search for optimal parameters
- Document parameter choices and rationale

### 3. Result Interpretation
- Consider attack success in context
- Analyze perturbation magnitude
- Look for patterns in successful attacks
- Use results to improve model robustness

### 4. Security Considerations
- Run attacks in isolated environments
- Monitor system resources during execution
- Log all attack activities
- Implement proper access controls

## Troubleshooting

### Common Issues

#### 1. Attack Execution Failures
- **Problem**: Attack fails to execute
- **Solutions**:
  - Check model compatibility
  - Verify input data format
  - Review attack parameters
  - Check system resources

#### 2. Low Success Rates
- **Problem**: Attacks have low success rates
- **Solutions**:
  - Increase perturbation magnitude
  - Adjust attack parameters
  - Try different attack types
  - Check model robustness

#### 3. High Computational Cost
- **Problem**: Attacks take too long to execute
- **Solutions**:
  - Reduce attack iterations
  - Use smaller datasets
  - Implement parallel execution
  - Use more efficient attacks

#### 4. Memory Issues
- **Problem**: Out of memory errors
- **Solutions**:
  - Reduce batch size
  - Use data streaming
  - Increase system memory
  - Optimize attack algorithms

## Support

For questions about specific attacks:
- Check attack documentation
- Review example code
- Contact support team
- Submit issue on GitHub
