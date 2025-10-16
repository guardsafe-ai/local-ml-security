"""
Database Repositories for Analytics Service
"""

import logging
from typing import Dict, List, Optional, Any
from .connection import db_manager
from models.requests import RedTeamTestResult, ModelPerformance

logger = logging.getLogger(__name__)


class RedTeamRepository:
    """Repository for red team test data"""
    
    async def store_test_result(self, result: RedTeamTestResult) -> Dict[str, str]:
        """Store red team test result with proper foreign key validation"""
        try:
            async with db_manager.transaction() as conn:
                # Insert main test result and verify success using RETURNING clause
                insert_result = await conn.fetchrow("""
                    INSERT INTO analytics.red_team_tests 
                    (test_id, model_name, model_type, model_version, model_source, 
                     total_attacks, vulnerabilities_found, detection_rate, 
                     test_duration_seconds, batch_size, attack_categories)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (test_id) DO UPDATE SET
                        model_name = EXCLUDED.model_name,
                        model_type = EXCLUDED.model_type,
                        model_version = EXCLUDED.model_version,
                        model_source = EXCLUDED.model_source,
                        total_attacks = EXCLUDED.total_attacks,
                        vulnerabilities_found = EXCLUDED.vulnerabilities_found,
                        detection_rate = EXCLUDED.detection_rate,
                        test_duration_seconds = EXCLUDED.test_duration_seconds,
                        batch_size = EXCLUDED.batch_size,
                        attack_categories = EXCLUDED.attack_categories
                    RETURNING test_id
                """, 
                result.test_id, result.model_name, result.model_type, 
                result.model_version, result.model_source,
                result.total_attacks, result.vulnerabilities_found, 
                result.detection_rate, result.test_duration_seconds,
                result.batch_size, result.attack_categories
                )
                
                # Verify parent record was inserted successfully
                if not insert_result or not insert_result['test_id']:
                    raise ValueError(f"Failed to insert test result for test_id: {result.test_id}")
                
                logger.info(f"✅ Successfully inserted/updated test result: {result.test_id}")
                
                # Now safe to insert child records - foreign key constraint is satisfied
                if result.attack_results:
                    for attack in result.attack_results:
                        await conn.execute("""
                            INSERT INTO analytics.attack_results 
                            (test_id, attack_pattern, attack_category, severity, 
                             model_prediction, confidence, detected, response_time_ms)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """, 
                        result.test_id,
                        attack.get('pattern', ''),
                        attack.get('category', ''),
                        attack.get('severity', 0.0),
                        attack.get('prediction', ''),
                        attack.get('confidence', 0.0),
                        attack.get('detected', False),
                        attack.get('response_time_ms', None)
                        )
                    
                    logger.info(f"✅ Successfully inserted {len(result.attack_results)} attack results for test_id: {result.test_id}")
                
                return {"message": "Red team results stored successfully", "test_id": result.test_id}
                
        except Exception as e:
            logger.error(f"Error storing results: {e}")
            raise Exception(f"Error storing results: {str(e)}")
    
    async def get_summary(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get red team test summary"""
        try:
            results = await db_manager.execute_query("""
                SELECT 
                    model_name,
                    model_type,
                    COUNT(*) as total_tests,
                    AVG(detection_rate) as avg_detection_rate,
                    AVG(total_attacks) as avg_attacks,
                    AVG(vulnerabilities_found) as avg_vulnerabilities,
                    MAX(test_timestamp) as last_test
                FROM analytics.red_team_tests 
                WHERE test_timestamp >= NOW() - INTERVAL '1 day' * $1
                GROUP BY model_name, model_type
                ORDER BY model_name, model_type
            """, days)
            
            return results
                
        except Exception as e:
            logger.error(f"Error fetching summary: {e}")
            raise Exception(f"Error fetching summary: {str(e)}")
    
    async def get_model_comparison(self, model_name: str, days: int = 30) -> Dict[str, Any]:
        """Get model comparison data"""
        try:
            # Get pre-trained model results
            pretrained_results = await db_manager.execute_query("""
                SELECT 
                    AVG(detection_rate) as avg_detection_rate,
                    AVG(total_attacks) as avg_attacks,
                    AVG(vulnerabilities_found) as avg_vulnerabilities,
                    COUNT(*) as test_count
                FROM analytics.red_team_tests 
                WHERE model_name = $1 
                AND model_type = 'pre-trained'
                AND test_timestamp >= NOW() - INTERVAL '1 day' * $2
            """, model_name, days)
            
            pretrained_stats = pretrained_results[0] if pretrained_results else None
            
            # Get trained model results
            trained_results = await db_manager.execute_query("""
                SELECT 
                    AVG(detection_rate) as avg_detection_rate,
                    AVG(total_attacks) as avg_attacks,
                    AVG(vulnerabilities_found) as avg_vulnerabilities,
                    COUNT(*) as test_count
                FROM analytics.red_team_tests 
                WHERE model_name = $1 
                AND model_type = 'trained'
                AND test_timestamp >= NOW() - INTERVAL '1 day' * $2
            """, model_name, days)
            
            trained_stats = trained_results[0] if trained_results else None
            
            # Calculate improvement
            improvement = {}
            if pretrained_stats and trained_stats:
                improvement = {
                    "detection_rate_improvement": float(trained_stats['avg_detection_rate'] or 0) - float(pretrained_stats['avg_detection_rate'] or 0),
                    "vulnerability_detection_improvement": float(trained_stats['avg_vulnerabilities'] or 0) - float(pretrained_stats['avg_vulnerabilities'] or 0)
                }
            
            return {
                "model_name": model_name,
                "pretrained": pretrained_stats,
                "trained": trained_stats,
                "improvement": improvement
            }
                
        except Exception as e:
            logger.error(f"Error fetching comparison: {e}")
            raise Exception(f"Error fetching comparison: {str(e)}")


class ModelPerformanceRepository:
    """Repository for model performance data"""
    
    async def store_performance(self, performance: ModelPerformance) -> Dict[str, str]:
        """Store model performance metrics with validation"""
        try:
            # Use transaction and RETURNING for verification
            async with db_manager.transaction() as conn:
                result = await conn.fetchrow("""
                    INSERT INTO analytics.model_performance 
                    (model_name, model_type, model_version, accuracy, precision, 
                     recall, f1_score, training_duration_seconds, dataset_size)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING id
                """, 
                performance.model_name, performance.model_type, performance.model_version,
                performance.accuracy, performance.precision, performance.recall,
                performance.f1_score, performance.training_duration_seconds, performance.dataset_size
                )
                
                if not result or not result['id']:
                    raise ValueError(f"Failed to insert performance metrics for model: {performance.model_name}")
                
                logger.info(f"✅ Successfully stored performance metrics for model: {performance.model_name}")
                
            return {"message": "Model performance stored successfully"}
                
        except Exception as e:
            logger.error(f"Error storing performance: {e}")
            raise Exception(f"Error storing performance: {str(e)}")


class AnalyticsRepository:
    """Repository for analytics queries"""
    
    async def get_performance_trends(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get performance trends over time"""
        try:
            results = await db_manager.execute_query("""
                SELECT 
                    DATE(test_timestamp) as test_date,
                    model_name,
                    model_type,
                    AVG(detection_rate) as avg_detection_rate,
                    COUNT(*) as test_count
                FROM analytics.red_team_tests 
                WHERE test_timestamp >= NOW() - INTERVAL '1 day' * $1
                GROUP BY DATE(test_timestamp), model_name, model_type
                ORDER BY test_date, model_name, model_type
            """, days)
            
            return results
                
        except Exception as e:
            logger.error(f"Error fetching trends: {e}")
            raise Exception(f"Error fetching trends: {str(e)}")
