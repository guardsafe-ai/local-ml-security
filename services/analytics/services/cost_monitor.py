"""
Cost Monitoring Service
Implements cost monitoring and alerts for cloud resources
"""

import logging
import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class CostCategory(Enum):
    """Cost categories"""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    ML_SERVICES = "ml_services"

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class CostAlert:
    """Cost alert model"""
    alert_id: str
    category: CostCategory
    severity: AlertSeverity
    message: str
    current_cost: float
    threshold: float
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False

@dataclass
class ResourceCost:
    """Resource cost model"""
    resource_id: str
    resource_type: str
    category: CostCategory
    hourly_cost: float
    daily_cost: float
    monthly_cost: float
    usage_hours: float
    last_updated: datetime

class CostMonitor:
    """Monitors and tracks cloud resource costs"""
    
    def __init__(self):
        self.resource_costs = {}
        self.cost_alerts = []
        self.budgets = {
            CostCategory.COMPUTE: {"daily": 50.0, "monthly": 1000.0},
            CostCategory.STORAGE: {"daily": 10.0, "monthly": 200.0},
            CostCategory.NETWORK: {"daily": 5.0, "monthly": 100.0},
            CostCategory.DATABASE: {"daily": 20.0, "monthly": 400.0},
            CostCategory.ML_SERVICES: {"daily": 30.0, "monthly": 600.0}
        }
        self.cost_history = []
        
    async def track_resource_cost(self, resource_id: str, resource_type: str,
                                category: CostCategory, usage_hours: float = 1.0) -> ResourceCost:
        """
        Track cost for a resource
        
        Args:
            resource_id: Unique identifier for the resource
            resource_type: Type of resource (instance, storage, etc.)
            category: Cost category
            usage_hours: Hours of usage
            
        Returns:
            Resource cost information
        """
        try:
            # Calculate costs based on resource type and category
            hourly_cost = self._calculate_hourly_cost(resource_type, category)
            daily_cost = hourly_cost * usage_hours
            monthly_cost = daily_cost * 30
            
            resource_cost = ResourceCost(
                resource_id=resource_id,
                resource_type=resource_type,
                category=category,
                hourly_cost=hourly_cost,
                daily_cost=daily_cost,
                monthly_cost=monthly_cost,
                usage_hours=usage_hours,
                last_updated=datetime.now(timezone.utc)
            )
            
            self.resource_costs[resource_id] = resource_cost
            
            # Add to cost history
            self.cost_history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "resource_id": resource_id,
                "category": category.value,
                "cost": daily_cost
            })
            
            logger.info(f"💰 [COST_MONITOR] Tracked cost for {resource_id}: ${daily_cost:.2f}/day")
            return resource_cost
            
        except Exception as e:
            logger.error(f"❌ [COST_MONITOR] Failed to track resource cost: {e}")
            raise
    
    def _calculate_hourly_cost(self, resource_type: str, category: CostCategory) -> float:
        """Calculate hourly cost for a resource"""
        # Simplified cost calculation (in real implementation, use cloud provider APIs)
        base_costs = {
            CostCategory.COMPUTE: {
                "cpu": 0.05,      # $0.05/hour per CPU core
                "gpu": 0.50,      # $0.50/hour per GPU
                "instance": 0.10  # $0.10/hour per instance
            },
            CostCategory.STORAGE: {
                "disk": 0.10,     # $0.10/GB/month
                "object": 0.05,   # $0.05/GB/month
                "backup": 0.02    # $0.02/GB/month
            },
            CostCategory.NETWORK: {
                "bandwidth": 0.01,  # $0.01/GB
                "load_balancer": 0.02  # $0.02/hour
            },
            CostCategory.DATABASE: {
                "postgres": 0.15,   # $0.15/hour
                "redis": 0.05,      # $0.05/hour
                "minio": 0.08       # $0.08/hour
            },
            CostCategory.ML_SERVICES: {
                "mlflow": 0.20,     # $0.20/hour
                "training": 0.30,   # $0.30/hour
                "inference": 0.15   # $0.15/hour
            }
        }
        
        category_costs = base_costs.get(category, {})
        return category_costs.get(resource_type, 0.05)  # Default cost
    
    async def get_daily_costs(self, date: datetime = None) -> Dict[str, Any]:
        """Get daily costs for all categories"""
        try:
            if date is None:
                date = datetime.now(timezone.utc)
            
            daily_costs = {}
            total_daily_cost = 0.0
            
            for category in CostCategory:
                category_cost = 0.0
                
                # Sum costs for resources in this category
                for resource_cost in self.resource_costs.values():
                    if resource_cost.category == category:
                        category_cost += resource_cost.daily_cost
                
                daily_costs[category.value] = {
                    "cost": category_cost,
                    "budget": self.budgets[category]["daily"],
                    "utilization_percent": (category_cost / self.budgets[category]["daily"]) * 100 if self.budgets[category]["daily"] > 0 else 0
                }
                
                total_daily_cost += category_cost
            
            daily_costs["total"] = {
                "cost": total_daily_cost,
                "budget": sum(budget["daily"] for budget in self.budgets.values()),
                "utilization_percent": (total_daily_cost / sum(budget["daily"] for budget in self.budgets.values())) * 100
            }
            
            return daily_costs
            
        except Exception as e:
            logger.error(f"❌ [COST_MONITOR] Failed to get daily costs: {e}")
            return {"error": str(e)}
    
    async def get_monthly_costs(self, month: int = None, year: int = None) -> Dict[str, Any]:
        """Get monthly costs for all categories"""
        try:
            if month is None:
                month = datetime.now().month
            if year is None:
                year = datetime.now().year
            
            monthly_costs = {}
            total_monthly_cost = 0.0
            
            for category in CostCategory:
                category_cost = 0.0
                
                # Sum costs for resources in this category
                for resource_cost in self.resource_costs.values():
                    if resource_cost.category == category:
                        category_cost += resource_cost.monthly_cost
                
                monthly_costs[category.value] = {
                    "cost": category_cost,
                    "budget": self.budgets[category]["monthly"],
                    "utilization_percent": (category_cost / self.budgets[category]["monthly"]) * 100 if self.budgets[category]["monthly"] > 0 else 0
                }
                
                total_monthly_cost += category_cost
            
            monthly_costs["total"] = {
                "cost": total_monthly_cost,
                "budget": sum(budget["monthly"] for budget in self.budgets.values()),
                "utilization_percent": (total_monthly_cost / sum(budget["monthly"] for budget in self.budgets.values())) * 100
            }
            
            return monthly_costs
            
        except Exception as e:
            logger.error(f"❌ [COST_MONITOR] Failed to get monthly costs: {e}")
            return {"error": str(e)}
    
    async def check_cost_alerts(self) -> List[CostAlert]:
        """Check for cost threshold violations and generate alerts"""
        try:
        new_alerts = []
        
        # Check daily costs
        daily_costs = await self.get_daily_costs()
        
        for category_str, cost_info in daily_costs.items():
            if category_str == "total":
                continue
            
            category = CostCategory(category_str)
            current_cost = cost_info["cost"]
            budget = cost_info["budget"]
            utilization_percent = cost_info["utilization_percent"]
            
            # Generate alerts based on utilization
            if utilization_percent >= 100:
                severity = AlertSeverity.CRITICAL
                message = f"Daily budget exceeded for {category.value}: ${current_cost:.2f} (${budget:.2f} budget)"
            elif utilization_percent >= 80:
                severity = AlertSeverity.HIGH
                message = f"Daily budget 80% exceeded for {category.value}: ${current_cost:.2f} (${budget:.2f} budget)"
            elif utilization_percent >= 60:
                severity = AlertSeverity.MEDIUM
                message = f"Daily budget 60% exceeded for {category.value}: ${current_cost:.2f} (${budget:.2f} budget)"
            elif utilization_percent >= 40:
                severity = AlertSeverity.LOW
                message = f"Daily budget 40% exceeded for {category.value}: ${current_cost:.2f} (${budget:.2f} budget)"
            else:
                continue
            
            # Check if alert already exists
            existing_alert = None
            for alert in self.cost_alerts:
                if (alert.category == category and 
                    alert.severity == severity and 
                    not alert.resolved):
                    existing_alert = alert
                    break
            
            if not existing_alert:
                alert = CostAlert(
                    alert_id=f"cost_{category.value}_{int(time.time())}",
                    category=category,
                    severity=severity,
                    message=message,
                    current_cost=current_cost,
                    threshold=budget,
                    timestamp=datetime.now(timezone.utc)
                )
                
                new_alerts.append(alert)
                self.cost_alerts.append(alert)
                
                logger.warning(f"🚨 [COST_MONITOR] {severity.value.upper()} alert: {message}")
        
        return new_alerts
        
        except Exception as e:
            logger.error(f"❌ [COST_MONITOR] Failed to check cost alerts: {e}")
            return []
    
    async def get_cost_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get cost trends over time"""
        try:
            trends = {
                "daily_costs": [],
                "category_trends": {},
                "total_trend": 0.0,
                "growth_rate": 0.0
            }
            
            # Calculate daily costs for the past N days
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            daily_totals = {}
            for category in CostCategory:
                daily_totals[category.value] = []
            
            # Simulate daily cost data (in real implementation, query historical data)
            for i in range(days):
                date = start_date + timedelta(days=i)
                daily_cost = 0.0
                
                for category in CostCategory:
                    # Simulate daily cost with some variation
                    base_cost = self.budgets[category]["daily"] * 0.3  # 30% of budget
                    variation = (i % 7) * 0.1  # Weekly pattern
                    category_daily_cost = base_cost + variation
                    
                    daily_totals[category.value].append(category_daily_cost)
                    daily_cost += category_daily_cost
                
                trends["daily_costs"].append({
                    "date": date.isoformat(),
                    "total_cost": daily_cost
                })
            
            # Calculate category trends
            for category in CostCategory:
                costs = daily_totals[category.value]
                if len(costs) >= 2:
                    growth_rate = ((costs[-1] - costs[0]) / costs[0]) * 100 if costs[0] > 0 else 0
                else:
                    growth_rate = 0
                
                trends["category_trends"][category.value] = {
                    "average_daily_cost": sum(costs) / len(costs),
                    "growth_rate": growth_rate,
                    "trend": "increasing" if growth_rate > 5 else "decreasing" if growth_rate < -5 else "stable"
                }
            
            # Calculate total trend
            total_costs = [day["total_cost"] for day in trends["daily_costs"]]
            if len(total_costs) >= 2:
                trends["total_trend"] = ((total_costs[-1] - total_costs[0]) / total_costs[0]) * 100 if total_costs[0] > 0 else 0
                trends["growth_rate"] = trends["total_trend"]
            
            return trends
            
        except Exception as e:
            logger.error(f"❌ [COST_MONITOR] Failed to get cost trends: {e}")
            return {"error": str(e)}
    
    async def get_cost_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get cost optimization recommendations"""
        try:
            recommendations = []
            
            # Analyze current costs
            daily_costs = await self.get_daily_costs()
            
            for category_str, cost_info in daily_costs.items():
                if category_str == "total":
                    continue
                
                category = CostCategory(category_str)
                utilization_percent = cost_info["utilization_percent"]
                
                if utilization_percent > 80:
                    recommendations.append({
                        "category": category.value,
                        "priority": "high",
                        "title": f"Optimize {category.value} costs",
                        "description": f"High utilization ({utilization_percent:.1f}%) suggests need for optimization",
                        "suggestions": self._get_optimization_suggestions(category),
                        "potential_savings": cost_info["cost"] * 0.2  # Assume 20% savings
                    })
                
                elif utilization_percent > 60:
                    recommendations.append({
                        "category": category.value,
                        "priority": "medium",
                        "title": f"Monitor {category.value} costs",
                        "description": f"Moderate utilization ({utilization_percent:.1f}%) - consider optimization",
                        "suggestions": self._get_optimization_suggestions(category),
                        "potential_savings": cost_info["cost"] * 0.1  # Assume 10% savings
                    })
            
            # Add general recommendations
            recommendations.extend(self._get_general_recommendations())
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ [COST_MONITOR] Failed to get optimization recommendations: {e}")
            return []
    
    def _get_optimization_suggestions(self, category: CostCategory) -> List[str]:
        """Get optimization suggestions for a category"""
        suggestions = {
            CostCategory.COMPUTE: [
                "Use spot instances for training jobs",
                "Implement auto-scaling to reduce idle instances",
                "Right-size instances based on actual usage",
                "Use reserved instances for predictable workloads"
            ],
            CostCategory.STORAGE: [
                "Implement data lifecycle policies",
                "Use compression for stored data",
                "Archive old data to cheaper storage tiers",
                "Clean up unused data regularly"
            ],
            CostCategory.NETWORK: [
                "Optimize data transfer patterns",
                "Use CDN for static content",
                "Implement request caching",
                "Monitor bandwidth usage"
            ],
            CostCategory.DATABASE: [
                "Optimize database queries",
                "Use connection pooling",
                "Implement read replicas",
                "Monitor and optimize indexes"
            ],
            CostCategory.ML_SERVICES: [
                "Use model quantization to reduce memory",
                "Implement model caching",
                "Batch inference requests",
                "Use ONNX for faster inference"
            ]
        }
        
        return suggestions.get(category, ["Review usage patterns and optimize"])
    
    def _get_general_recommendations(self) -> List[Dict[str, Any]]:
        """Get general cost optimization recommendations"""
        return [
            {
                "category": "general",
                "priority": "medium",
                "title": "Implement cost monitoring",
                "description": "Set up automated cost monitoring and alerts",
                "suggestions": [
                    "Set up daily cost alerts",
                    "Implement budget notifications",
                    "Create cost dashboards",
                    "Schedule regular cost reviews"
                ],
                "potential_savings": 0
            },
            {
                "category": "general",
                "priority": "low",
                "title": "Use cost allocation tags",
                "description": "Implement proper cost allocation and tagging",
                "suggestions": [
                    "Tag all resources with project/environment",
                    "Implement cost center allocation",
                    "Track costs by team/department",
                    "Set up cost reporting by tags"
                ],
                "potential_savings": 0
            }
        ]
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a cost alert"""
        try:
            for alert in self.cost_alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    logger.info(f"✅ [COST_MONITOR] Acknowledged alert: {alert_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ [COST_MONITOR] Failed to acknowledge alert: {e}")
            return False
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve a cost alert"""
        try:
            for alert in self.cost_alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    logger.info(f"✅ [COST_MONITOR] Resolved alert: {alert_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ [COST_MONITOR] Failed to resolve alert: {e}")
            return False
    
    def update_budget(self, category: CostCategory, daily_budget: float, monthly_budget: float):
        """Update budget for a category"""
        self.budgets[category] = {
            "daily": daily_budget,
            "monthly": monthly_budget
        }
        logger.info(f"✅ [COST_MONITOR] Updated budget for {category.value}: ${daily_budget}/day, ${monthly_budget}/month")

# Global cost monitor
cost_monitor = CostMonitor()
