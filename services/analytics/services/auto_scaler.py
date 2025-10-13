"""
Auto-Scaling Service
Implements horizontal pod autoscaling and load-based scaling
"""

import logging
import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import httpx

logger = logging.getLogger(__name__)

class ScalingTrigger(Enum):
    """Scaling trigger types"""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"

class ScalingAction(Enum):
    """Scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"

@dataclass
class ScalingRule:
    """Scaling rule configuration"""
    trigger: ScalingTrigger
    threshold: float
    action: ScalingAction
    cooldown_period: int = 300  # seconds
    min_instances: int = 1
    max_instances: int = 10
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.7

@dataclass
class ServiceInstance:
    """Service instance model"""
    instance_id: str
    service_name: str
    status: str
    cpu_usage: float
    memory_usage: float
    request_count: int
    response_time: float
    last_updated: datetime

class AutoScaler:
    """Handles auto-scaling of services based on metrics"""
    
    def __init__(self):
        self.services = {
            "model-api": {
                "current_instances": 1,
                "target_instances": 1,
                "min_instances": 1,
                "max_instances": 5,
                "scaling_rules": [
                    ScalingRule(ScalingTrigger.CPU_UTILIZATION, 70.0, ScalingAction.SCALE_UP),
                    ScalingRule(ScalingTrigger.CPU_UTILIZATION, 30.0, ScalingAction.SCALE_DOWN),
                    ScalingRule(ScalingTrigger.REQUEST_RATE, 100.0, ScalingAction.SCALE_UP),
                    ScalingRule(ScalingTrigger.RESPONSE_TIME, 2.0, ScalingAction.SCALE_UP)
                ]
            },
            "training": {
                "current_instances": 1,
                "target_instances": 1,
                "min_instances": 1,
                "max_instances": 3,
                "scaling_rules": [
                    ScalingRule(ScalingTrigger.QUEUE_LENGTH, 5.0, ScalingAction.SCALE_UP),
                    ScalingRule(ScalingTrigger.CPU_UTILIZATION, 80.0, ScalingAction.SCALE_UP)
                ]
            },
            "analytics": {
                "current_instances": 1,
                "target_instances": 1,
                "min_instances": 1,
                "max_instances": 3,
                "scaling_rules": [
                    ScalingRule(ScalingTrigger.CPU_UTILIZATION, 75.0, ScalingAction.SCALE_UP),
                    ScalingRule(ScalingTrigger.MEMORY_UTILIZATION, 80.0, ScalingAction.SCALE_UP)
                ]
            }
        }
        
        self.instances = {}
        self.scaling_history = []
        self.last_scaling_action = {}
        
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all services"""
        try:
            metrics = {}
            
            for service_name in self.services.keys():
                service_metrics = await self._collect_service_metrics(service_name)
                metrics[service_name] = service_metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ [AUTO_SCALER] Failed to collect metrics: {e}")
            return {}
    
    async def _collect_service_metrics(self, service_name: str) -> Dict[str, Any]:
        """Collect metrics for a specific service"""
        try:
            # Get service instances
            instances = await self._get_service_instances(service_name)
            
            if not instances:
                return {"error": "No instances found"}
            
            # Calculate aggregated metrics
            total_cpu = sum(instance.get("cpu_usage", 0) for instance in instances)
            total_memory = sum(instance.get("memory_usage", 0) for instance in instances)
            total_requests = sum(instance.get("request_count", 0) for instance in instances)
            total_response_time = sum(instance.get("response_time", 0) for instance in instances)
            
            instance_count = len(instances)
            
            metrics = {
                "instance_count": instance_count,
                "avg_cpu_usage": total_cpu / instance_count if instance_count > 0 else 0,
                "avg_memory_usage": total_memory / instance_count if instance_count > 0 else 0,
                "total_requests": total_requests,
                "avg_response_time": total_response_time / instance_count if instance_count > 0 else 0,
                "instances": instances
            }
            
            # Add queue length for training service
            if service_name == "training":
                queue_length = await self._get_training_queue_length()
                metrics["queue_length"] = queue_length
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ [AUTO_SCALER] Failed to collect metrics for {service_name}: {e}")
            return {"error": str(e)}
    
    async def _get_service_instances(self, service_name: str) -> List[Dict[str, Any]]:
        """Get instances for a service"""
        try:
            # In a real implementation, this would query Kubernetes or Docker
            # For now, we'll simulate based on current instances
            
            current_instances = self.services[service_name]["current_instances"]
            instances = []
            
            for i in range(current_instances):
                instance_id = f"{service_name}-{i}"
                
                # Simulate metrics (in real implementation, get from monitoring system)
                instance_metrics = {
                    "instance_id": instance_id,
                    "cpu_usage": self._simulate_cpu_usage(service_name),
                    "memory_usage": self._simulate_memory_usage(service_name),
                    "request_count": self._simulate_request_count(service_name),
                    "response_time": self._simulate_response_time(service_name),
                    "status": "healthy"
                }
                
                instances.append(instance_metrics)
            
            return instances
            
        except Exception as e:
            logger.error(f"❌ [AUTO_SCALER] Failed to get instances for {service_name}: {e}")
            return []
    
    def _simulate_cpu_usage(self, service_name: str) -> float:
        """Simulate CPU usage (replace with real metrics)"""
        # Simulate different CPU patterns for different services
        base_usage = {
            "model-api": 45.0,
            "training": 25.0,
            "analytics": 35.0
        }
        
        # Add some randomness
        import random
        return base_usage.get(service_name, 30.0) + random.uniform(-10, 10)
    
    def _simulate_memory_usage(self, service_name: str) -> float:
        """Simulate memory usage (replace with real metrics)"""
        base_usage = {
            "model-api": 60.0,
            "training": 40.0,
            "analytics": 50.0
        }
        
        import random
        return base_usage.get(service_name, 40.0) + random.uniform(-5, 5)
    
    def _simulate_request_count(self, service_name: str) -> int:
        """Simulate request count (replace with real metrics)"""
        base_count = {
            "model-api": 150,
            "training": 5,
            "analytics": 25
        }
        
        import random
        return base_count.get(service_name, 10) + random.randint(-20, 20)
    
    def _simulate_response_time(self, service_name: str) -> float:
        """Simulate response time (replace with real metrics)"""
        base_time = {
            "model-api": 0.5,
            "training": 30.0,
            "analytics": 1.2
        }
        
        import random
        return base_time.get(service_name, 1.0) + random.uniform(-0.2, 0.2)
    
    async def _get_training_queue_length(self) -> int:
        """Get training queue length"""
        try:
            # In real implementation, query the training queue service
            return 3  # Simulated
        except Exception as e:
            logger.error(f"❌ [AUTO_SCALER] Failed to get queue length: {e}")
            return 0
    
    async def evaluate_scaling(self) -> Dict[str, Any]:
        """Evaluate scaling decisions for all services"""
        try:
            scaling_decisions = {}
            metrics = await self.collect_metrics()
            
            for service_name, service_config in self.services.items():
                service_metrics = metrics.get(service_name, {})
                if "error" in service_metrics:
                    continue
                
                decision = await self._evaluate_service_scaling(service_name, service_config, service_metrics)
                scaling_decisions[service_name] = decision
            
            return scaling_decisions
            
        except Exception as e:
            logger.error(f"❌ [AUTO_SCALER] Failed to evaluate scaling: {e}")
            return {}
    
    async def _evaluate_service_scaling(self, service_name: str, service_config: Dict[str, Any], 
                                      metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate scaling for a specific service"""
        try:
            current_instances = service_config["current_instances"]
            target_instances = current_instances
            action_taken = ScalingAction.NO_ACTION
            reason = "No scaling needed"
            
            # Check cooldown period
            last_action_time = self.last_scaling_action.get(service_name, 0)
            if time.time() - last_action_time < 300:  # 5 minute cooldown
                return {
                    "service": service_name,
                    "current_instances": current_instances,
                    "target_instances": target_instances,
                    "action": ScalingAction.NO_ACTION.value,
                    "reason": "Cooldown period active"
                }
            
            # Evaluate each scaling rule
            for rule in service_config["scaling_rules"]:
                trigger_value = self._get_trigger_value(rule.trigger, metrics)
                
                if trigger_value is None:
                    continue
                
                should_scale = False
                
                if rule.action == ScalingAction.SCALE_UP:
                    if trigger_value > rule.threshold:
                        should_scale = True
                        reason = f"{rule.trigger.value} ({trigger_value:.1f}) > threshold ({rule.threshold})"
                
                elif rule.action == ScalingAction.SCALE_DOWN:
                    if trigger_value < rule.threshold:
                        should_scale = True
                        reason = f"{rule.trigger.value} ({trigger_value:.1f}) < threshold ({rule.threshold})"
                
                if should_scale:
                    if rule.action == ScalingAction.SCALE_UP:
                        target_instances = min(
                            int(current_instances * rule.scale_up_factor),
                            service_config["max_instances"]
                        )
                    else:
                        target_instances = max(
                            int(current_instances * rule.scale_down_factor),
                            service_config["min_instances"]
                        )
                    
                    action_taken = rule.action
                    break
            
            # Record scaling decision
            if action_taken != ScalingAction.NO_ACTION:
                self.last_scaling_action[service_name] = time.time()
                
                self.scaling_history.append({
                    "service": service_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "action": action_taken.value,
                    "from_instances": current_instances,
                    "to_instances": target_instances,
                    "reason": reason
                })
            
            return {
                "service": service_name,
                "current_instances": current_instances,
                "target_instances": target_instances,
                "action": action_taken.value,
                "reason": reason,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"❌ [AUTO_SCALER] Failed to evaluate scaling for {service_name}: {e}")
            return {"error": str(e)}
    
    def _get_trigger_value(self, trigger: ScalingTrigger, metrics: Dict[str, Any]) -> Optional[float]:
        """Get trigger value from metrics"""
        try:
            if trigger == ScalingTrigger.CPU_UTILIZATION:
                return metrics.get("avg_cpu_usage")
            elif trigger == ScalingTrigger.MEMORY_UTILIZATION:
                return metrics.get("avg_memory_usage")
            elif trigger == ScalingTrigger.REQUEST_RATE:
                return metrics.get("total_requests", 0) / 60  # requests per minute
            elif trigger == ScalingTrigger.QUEUE_LENGTH:
                return metrics.get("queue_length", 0)
            elif trigger == ScalingTrigger.RESPONSE_TIME:
                return metrics.get("avg_response_time")
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ [AUTO_SCALER] Failed to get trigger value: {e}")
            return None
    
    async def execute_scaling(self, scaling_decisions: Dict[str, Any]) -> Dict[str, Any]:
        """Execute scaling decisions"""
        try:
            results = {}
            
            for service_name, decision in scaling_decisions.items():
                if "error" in decision:
                    continue
                
                current_instances = decision["current_instances"]
                target_instances = decision["target_instances"]
                action = decision["action"]
                
                if action == ScalingAction.NO_ACTION.value:
                    results[service_name] = {"status": "no_action", "reason": decision["reason"]}
                    continue
                
                # Execute scaling
                success = await self._scale_service(service_name, target_instances)
                
                if success:
                    self.services[service_name]["current_instances"] = target_instances
                    results[service_name] = {
                        "status": "success",
                        "action": action,
                        "from_instances": current_instances,
                        "to_instances": target_instances,
                        "reason": decision["reason"]
                    }
                    logger.info(f"✅ [AUTO_SCALER] Scaled {service_name}: {current_instances} -> {target_instances}")
                else:
                    results[service_name] = {
                        "status": "failed",
                        "action": action,
                        "reason": "Scaling execution failed"
                    }
                    logger.error(f"❌ [AUTO_SCALER] Failed to scale {service_name}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ [AUTO_SCALER] Failed to execute scaling: {e}")
            return {"error": str(e)}
    
    async def _scale_service(self, service_name: str, target_instances: int) -> bool:
        """Scale a service to target number of instances"""
        try:
            # In a real implementation, this would:
            # 1. Call Kubernetes API to scale deployment
            # 2. Or call Docker Swarm API
            # 3. Or call cloud provider API (AWS ECS, GCP Cloud Run, etc.)
            
            # For simulation, we'll just update the configuration
            logger.info(f"🔄 [AUTO_SCALER] Scaling {service_name} to {target_instances} instances")
            
            # Simulate scaling delay
            await asyncio.sleep(1)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ [AUTO_SCALER] Failed to scale service {service_name}: {e}")
            return False
    
    async def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status"""
        try:
            status = {
                "services": {},
                "scaling_history": self.scaling_history[-10:],  # Last 10 scaling actions
                "total_scaling_actions": len(self.scaling_history)
            }
            
            for service_name, service_config in self.services.items():
                status["services"][service_name] = {
                    "current_instances": service_config["current_instances"],
                    "target_instances": service_config["target_instances"],
                    "min_instances": service_config["min_instances"],
                    "max_instances": service_config["max_instances"],
                    "scaling_rules": [
                        {
                            "trigger": rule.trigger.value,
                            "threshold": rule.threshold,
                            "action": rule.action.value,
                            "cooldown_period": rule.cooldown_period
                        }
                        for rule in service_config["scaling_rules"]
                    ]
                }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ [AUTO_SCALER] Failed to get scaling status: {e}")
            return {"error": str(e)}
    
    async def run_scaling_loop(self):
        """Run the auto-scaling loop"""
        logger.info("🚀 [AUTO_SCALER] Starting auto-scaling loop")
        
        while True:
            try:
                # Evaluate scaling decisions
                scaling_decisions = await self.evaluate_scaling()
                
                # Execute scaling
                if scaling_decisions:
                    results = await self.execute_scaling(scaling_decisions)
                    logger.info(f"📊 [AUTO_SCALER] Scaling cycle completed: {results}")
                
                # Wait before next evaluation
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"❌ [AUTO_SCALER] Error in scaling loop: {e}")
                await asyncio.sleep(60)

# Global auto-scaler
auto_scaler = AutoScaler()
