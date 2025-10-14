"""
Background Tasks for Analytics Service
Scheduled drift detection and alerting
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)

class DriftMonitoringScheduler:
    """Scheduled drift detection and alerting"""
    
    def __init__(self, drift_detector, email_service, db_manager=None):
        self.drift_detector = drift_detector
        self.email_service = email_service
        self.db_manager = db_manager
        self.scheduler = AsyncIOScheduler()
        self.is_running = False
        
        logger.info("✅ DriftMonitoringScheduler initialized")
    
    async def start(self):
        """Start the drift monitoring scheduler"""
        if self.is_running:
            logger.warning("Drift monitoring scheduler is already running")
            return
        
        # Schedule drift checks every 6 hours
        self.scheduler.add_job(
            self.check_drift_for_production_models,
            trigger=IntervalTrigger(hours=6),
            id='drift_monitoring',
            name='Production Model Drift Monitoring',
            replace_existing=True
        )
        
        # Schedule daily drift reports
        self.scheduler.add_job(
            self.generate_daily_drift_report,
            trigger=IntervalTrigger(hours=24),
            id='daily_drift_report',
            name='Daily Drift Report',
            replace_existing=True
        )
        
        self.scheduler.start()
        self.is_running = True
        logger.info("🔄 Drift monitoring scheduler started (every 6 hours)")
    
    async def stop(self):
        """Stop the drift monitoring scheduler"""
        if not self.is_running:
            return
        
        self.scheduler.shutdown()
        self.is_running = False
        logger.info("⏹️ Drift monitoring scheduler stopped")
    
    async def check_drift_for_production_models(self):
        """Check drift for all production models"""
        logger.info("🔍 Starting scheduled drift detection...")
        
        try:
            # Get all production models
            production_models = await self.get_production_models()
            
            if not production_models:
                logger.info("ℹ️ No production models found for drift monitoring")
                return
            
            logger.info(f"📊 Monitoring {len(production_models)} production models for drift")
            
            drift_results = []
            for model in production_models:
                try:
                    model_name = model["name"]
                    logger.info(f"🔍 Checking drift for model: {model_name}")
                    
                    # Detect drift using production inference data
                    drift_result = await self.drift_detector.detect_data_drift_with_production_data(
                        model_name=model_name,
                        hours=24  # Last 24 hours of data
                    )
                    
                    if drift_result.get("drift_detected", False):
                        logger.warning(f"⚠️ Drift detected for {model_name}")
                        drift_results.append({
                            "model_name": model_name,
                            "drift_score": drift_result.get("drift_score", 0.0),
                            "drift_type": drift_result.get("drift_type", "unknown"),
                            "details": drift_result
                        })
                        
                        # Send email alert (email service already implemented!)
                        await self.email_service.send_drift_alert(
                            drift_results=drift_result,
                            model_name=model_name
                        )
                        
                        # Log drift detection event
                        await self.log_drift_event(
                            model_name=model_name,
                            drift_score=drift_result.get("drift_score", 0.0),
                            drift_type=drift_result.get("drift_type", "unknown"),
                            detected=True
                        )
                    else:
                        logger.info(f"✅ No drift detected for {model_name}")
                        
                        # Log successful check
                        await self.log_drift_event(
                            model_name=model_name,
                            drift_score=drift_result.get("drift_score", 0.0),
                            drift_type=drift_result.get("drift_type", "unknown"),
                            detected=False
                        )
                
                except Exception as e:
                    logger.error(f"❌ Error checking drift for {model['name']}: {e}")
                    continue
            
            # Generate summary
            if drift_results:
                logger.warning(f"🚨 Drift monitoring completed: {len(drift_results)} models with drift detected")
            else:
                logger.info("✅ Drift monitoring completed: No drift detected in any production models")
                
        except Exception as e:
            logger.error(f"❌ Error in scheduled drift detection: {e}")
    
    async def generate_daily_drift_report(self):
        """Generate daily drift monitoring report"""
        logger.info("📊 Generating daily drift report...")
        
        try:
            # Get drift statistics for the last 24 hours
            drift_stats = await self.get_drift_statistics(hours=24)
            
            # Generate report
            report = {
                "date": datetime.now().isoformat(),
                "total_models_monitored": drift_stats.get("total_models", 0),
                "drift_detections": drift_stats.get("drift_detections", 0),
                "models_with_drift": drift_stats.get("models_with_drift", []),
                "average_drift_score": drift_stats.get("average_drift_score", 0.0),
                "max_drift_score": drift_stats.get("max_drift_score", 0.0)
            }
            
            # Log report
            logger.info("📈 Daily drift report", extra={"report": report})
            
            # Store report in database if available
            if self.db_manager:
                await self.store_drift_report(report)
                
        except Exception as e:
            logger.error(f"❌ Error generating daily drift report: {e}")
    
    async def get_production_models(self) -> List[Dict[str, Any]]:
        """Get all production models"""
        try:
            # This would typically query MLflow or a model registry
            # For now, return a mock list
            return [
                {"name": "distilbert", "version": "v15", "stage": "Production"},
                {"name": "bert-base", "version": "v12", "stage": "Production"},
                {"name": "roberta-base", "version": "v8", "stage": "Production"}
            ]
        except Exception as e:
            logger.error(f"Error getting production models: {e}")
            return []
    
    async def get_drift_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get drift statistics for the specified time period"""
        try:
            if not self.db_manager:
                return {"total_models": 0, "drift_detections": 0}
            
            # Query drift detection results from the last N hours
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # This would query the database for drift statistics
            # For now, return mock data
            return {
                "total_models": 3,
                "drift_detections": 0,
                "models_with_drift": [],
                "average_drift_score": 0.0,
                "max_drift_score": 0.0
            }
        except Exception as e:
            logger.error(f"Error getting drift statistics: {e}")
            return {"total_models": 0, "drift_detections": 0}
    
    async def log_drift_event(
        self, 
        model_name: str, 
        drift_score: float, 
        drift_type: str, 
        detected: bool
    ):
        """Log drift detection event"""
        try:
            if self.db_manager:
                await self.db_manager.execute(
                    """
                    INSERT INTO drift_monitoring_log 
                    (model_name, drift_score, drift_type, detected, timestamp)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    model_name, drift_score, drift_type, detected, datetime.now()
                )
        except Exception as e:
            logger.error(f"Error logging drift event: {e}")
    
    async def store_drift_report(self, report: Dict[str, Any]):
        """Store daily drift report in database"""
        try:
            if self.db_manager:
                await self.db_manager.execute(
                    """
                    INSERT INTO drift_daily_reports 
                    (report_date, report_data, created_at)
                    VALUES ($1, $2, $3)
                    """,
                    report["date"], 
                    json.dumps(report), 
                    datetime.now()
                )
        except Exception as e:
            logger.error(f"Error storing drift report: {e}")

# Global scheduler instance
drift_scheduler = None

async def start_drift_monitoring(drift_detector, email_service, db_manager=None):
    """Start drift monitoring scheduler"""
    global drift_scheduler
    
    if drift_scheduler is None:
        drift_scheduler = DriftMonitoringScheduler(drift_detector, email_service, db_manager)
        await drift_scheduler.start()
        logger.info("🚀 Drift monitoring started")
    else:
        logger.warning("Drift monitoring is already running")

async def stop_drift_monitoring():
    """Stop drift monitoring scheduler"""
    global drift_scheduler
    
    if drift_scheduler:
        await drift_scheduler.stop()
        drift_scheduler = None
        logger.info("⏹️ Drift monitoring stopped")
