"""
Red Team Service - Database Repositories
Data access layer for red team service
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from database.connection import db_manager

logger = logging.getLogger(__name__)


class RedTeamRepository:
    """Repository for red team test operations"""
    
    async def save_test_result(self, test_id: str, model_name: str, model_type: str,
                             attack_category: str, attack_pattern: str, attack_severity: float,
                             detected: bool, confidence: float = None, response_time_ms: float = None,
                             test_duration_ms: float = None, vulnerability_score: float = None,
                             security_risk: str = None, pass_fail: bool = None) -> bool:
        """Save individual test result"""
        try:
            await db_manager.execute_command("""
                INSERT INTO red_team_test_results (
                    test_id, model_name, model_type, attack_category, attack_pattern,
                    attack_severity, detected, confidence, response_time_ms, test_duration_ms,
                    vulnerability_score, security_risk, pass_fail
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """, test_id, model_name, model_type, attack_category, attack_pattern,
                attack_severity, detected, confidence, response_time_ms, test_duration_ms,
                vulnerability_score, security_risk, pass_fail)
            return True
        except Exception as e:
            logger.error(f"Failed to save test result: {e}")
            return False

    async def save_test_session(self, session_id: str, model_name: str, model_type: str,
                              total_attacks: int, detected_attacks: int, detection_rate: float,
                              pass_rate: float, overall_status: str, start_time: datetime,
                              end_time: datetime = None, duration_ms: float = None) -> bool:
        """Save test session summary"""
        try:
            await db_manager.execute_command("""
                INSERT INTO red_team_test_sessions (
                    session_id, model_name, model_type, total_attacks, detected_attacks,
                    detection_rate, pass_rate, overall_status, start_time, end_time, duration_ms
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """, session_id, model_name, model_type, total_attacks, detected_attacks,
                detection_rate, pass_rate, overall_status, start_time, end_time, duration_ms)
            return True
        except Exception as e:
            logger.error(f"Failed to save test session: {e}")
            return False

    async def get_test_results(self, test_id: str) -> List[Dict[str, Any]]:
        """Get test results by test ID"""
        try:
            results = await db_manager.execute_query("""
                SELECT * FROM red_team_test_results 
                WHERE test_id = $1 
                ORDER BY timestamp DESC
            """, test_id)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to get test results for {test_id}: {e}")
            return []

    async def get_test_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get test session by session ID"""
        try:
            results = await db_manager.execute_query("""
                SELECT * FROM red_team_test_sessions 
                WHERE session_id = $1
            """, session_id)
            return dict(results[0]) if results else None
        except Exception as e:
            logger.error(f"Failed to get test session {session_id}: {e}")
            return None

    async def get_latest_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get latest test results"""
        try:
            results = await db_manager.execute_query("""
                SELECT * FROM red_team_test_results 
                ORDER BY timestamp DESC 
                LIMIT $1
            """, limit)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to get latest results: {e}")
            return []

    async def get_results_by_model(self, model_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get test results for a specific model"""
        try:
            results = await db_manager.execute_query("""
                SELECT * FROM red_team_test_results 
                WHERE model_name = $1 
                ORDER BY timestamp DESC 
                LIMIT $2
            """, model_name, limit)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to get results for model {model_name}: {e}")
            return []

    async def get_results_by_category(self, category: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get test results for a specific attack category"""
        try:
            results = await db_manager.execute_query("""
                SELECT * FROM red_team_test_results 
                WHERE attack_category = $1 
                ORDER BY timestamp DESC 
                LIMIT $2
            """, category, limit)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to get results for category {category}: {e}")
            return []

    async def get_metrics(self, time_range_hours: int = 24, model_name: str = None, 
                         category: str = None) -> Dict[str, Any]:
        """Get aggregated metrics"""
        try:
            # Build query based on filters
            where_conditions = ["timestamp >= %s"]
            params = [datetime.now() - timedelta(hours=time_range_hours)]
            
            if model_name:
                where_conditions.append("model_name = %s")
                params.append(model_name)
            
            if category:
                where_conditions.append("attack_category = %s")
                params.append(category)
            
            where_clause = " AND ".join(where_conditions)
            
            # Get basic metrics
            metrics = await db_manager.execute_query(f"""
                SELECT 
                    COUNT(*) as total_tests,
                    COUNT(CASE WHEN detected = true THEN 1 END) as detected_attacks,
                    AVG(CASE WHEN detected = true THEN 1.0 ELSE 0.0 END) as detection_rate,
                    AVG(confidence) as avg_confidence,
                    AVG(response_time_ms) as avg_response_time,
                    AVG(test_duration_ms) as avg_test_duration
                FROM red_team_test_results 
                WHERE {where_clause}
            """, *params)
            
            # Get category breakdown
            category_breakdown = await db_manager.execute_query(f"""
                SELECT 
                    attack_category,
                    COUNT(*) as total_attacks,
                    COUNT(CASE WHEN detected = true THEN 1 END) as detected_attacks,
                    AVG(CASE WHEN detected = true THEN 1.0 ELSE 0.0 END) as detection_rate,
                    AVG(confidence) as avg_confidence
                FROM red_team_test_results 
                WHERE {where_clause}
                GROUP BY attack_category
                ORDER BY total_attacks DESC
            """, *params)
            
            # Get model performance
            model_performance = await db_manager.execute_query(f"""
                SELECT 
                    model_name,
                    COUNT(*) as total_tests,
                    COUNT(CASE WHEN detected = true THEN 1 END) as detected_attacks,
                    AVG(CASE WHEN detected = true THEN 1.0 ELSE 0.0 END) as detection_rate,
                    AVG(confidence) as avg_confidence
                FROM red_team_test_results 
                WHERE {where_clause}
                GROUP BY model_name
                ORDER BY total_tests DESC
            """, *params)
            
            # Get risk distribution
            risk_distribution = await db_manager.execute_query(f"""
                SELECT 
                    security_risk,
                    COUNT(*) as count
                FROM red_team_test_results 
                WHERE {where_clause} AND security_risk IS NOT NULL
                GROUP BY security_risk
                ORDER BY count DESC
            """, *params)
            
            return {
                "metrics": dict(metrics[0]) if metrics else {},
                "category_breakdown": [dict(row) for row in category_breakdown],
                "model_performance": [dict(row) for row in model_performance],
                "risk_distribution": {row["security_risk"]: row["count"] for row in risk_distribution}
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {
                "metrics": {},
                "category_breakdown": [],
                "model_performance": [],
                "risk_distribution": {}
            }

    async def get_time_series_data(self, time_range_hours: int = 24, 
                                  model_name: str = None) -> List[Dict[str, Any]]:
        """Get time series data for charts"""
        try:
            where_conditions = ["timestamp >= %s"]
            params = [datetime.now() - timedelta(hours=time_range_hours)]
            
            if model_name:
                where_conditions.append("model_name = %s")
                params.append(model_name)
            
            where_clause = " AND ".join(where_conditions)
            
            results = await db_manager.execute_query(f"""
                SELECT 
                    DATE_TRUNC('hour', timestamp) as hour,
                    COUNT(*) as total_attacks,
                    COUNT(CASE WHEN detected = true THEN 1 END) as detected_attacks,
                    AVG(CASE WHEN detected = true THEN 1.0 ELSE 0.0 END) as detection_rate,
                    AVG(confidence) as avg_confidence
                FROM red_team_test_results 
                WHERE {where_clause}
                GROUP BY DATE_TRUNC('hour', timestamp)
                ORDER BY hour ASC
            """, *params)
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to get time series data: {e}")
            return []

    async def cleanup_old_data(self, days_old: int = 30) -> int:
        """Clean up old test data"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            # Clean up old test results
            results_deleted = await db_manager.execute_command("""
                DELETE FROM red_team_test_results 
                WHERE created_at < $1
            """, cutoff_date)
            
            # Clean up old test sessions
            sessions_deleted = await db_manager.execute_command("""
                DELETE FROM red_team_test_sessions 
                WHERE created_at < $1
            """, cutoff_date)
            
            total_deleted = results_deleted + sessions_deleted
            logger.info(f"Cleaned up {total_deleted} old records")
            return total_deleted
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0
