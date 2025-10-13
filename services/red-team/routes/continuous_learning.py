"""
Continuous Learning Routes
Provides endpoints for automated pattern learning and threat intelligence
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from services.continuous_learning import ContinuousLearningService, ThreatLevel, LearningType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/continuous-learning", tags=["continuous-learning"])

# Initialize continuous learning service
learning_service = ContinuousLearningService()


@router.post("/learn")
async def learn_from_data(attack_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Learn from new attack data
    
    Args:
        attack_data: New attack data to learn from
        
    Returns:
        Learning results
    """
    try:
        results = await learning_service.learn_from_attack_data(attack_data)
        return {
            "status": "success",
            "message": "Learning completed successfully",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Continuous learning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns")
async def get_learned_patterns(
    attack_type: Optional[str] = Query(None, description="Filter by attack type"),
    threat_level: Optional[str] = Query(None, description="Filter by threat level")
) -> Dict[str, Any]:
    """
    Get learned attack patterns with optional filtering
    
    Args:
        attack_type: Filter by attack type
        threat_level: Filter by threat level
        
    Returns:
        Learned patterns
    """
    try:
        threat_level_enum = None
        if threat_level:
            try:
                threat_level_enum = ThreatLevel(threat_level.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid threat level: {threat_level}")
        
        patterns = await learning_service.get_learned_patterns(attack_type, threat_level_enum)
        
        return {
            "status": "success",
            "patterns": patterns,
            "count": len(patterns),
            "filters": {
                "attack_type": attack_type,
                "threat_level": threat_level
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Patterns retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threats")
async def get_threat_intelligence(
    threat_type: Optional[str] = Query(None, description="Filter by threat type"),
    severity: Optional[str] = Query(None, description="Filter by severity level")
) -> Dict[str, Any]:
    """
    Get threat intelligence with optional filtering
    
    Args:
        threat_type: Filter by threat type
        severity: Filter by severity level
        
    Returns:
        Threat intelligence data
    """
    try:
        severity_enum = None
        if severity:
            try:
                severity_enum = ThreatLevel(severity.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid severity level: {severity}")
        
        threats = await learning_service.get_threat_intelligence(threat_type, severity_enum)
        
        return {
            "status": "success",
            "threats": threats,
            "count": len(threats),
            "filters": {
                "threat_type": threat_type,
                "severity": severity
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Threat intelligence retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/anomalies")
async def get_anomalies(
    anomaly_type: Optional[str] = Query(None, description="Filter by anomaly type"),
    severity: Optional[str] = Query(None, description="Filter by severity level")
) -> Dict[str, Any]:
    """
    Get detected anomalies with optional filtering
    
    Args:
        anomaly_type: Filter by anomaly type
        severity: Filter by severity level
        
    Returns:
        Detected anomalies
    """
    try:
        severity_enum = None
        if severity:
            try:
                severity_enum = ThreatLevel(severity.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid severity level: {severity}")
        
        anomalies = await learning_service.get_anomalies(anomaly_type, severity_enum)
        
        return {
            "status": "success",
            "anomalies": anomalies,
            "count": len(anomalies),
            "filters": {
                "anomaly_type": anomaly_type,
                "severity": severity
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Anomalies retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_learning_summary() -> Dict[str, Any]:
    """
    Get comprehensive learning summary
    
    Returns:
        Learning summary
    """
    try:
        summary = await learning_service.get_learning_summary()
        
        return {
            "status": "success",
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Learning summary retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-adaptive-attacks")
async def generate_adaptive_attacks(
    target_model: str,
    attack_count: int = Query(10, ge=1, le=100, description="Number of attacks to generate")
) -> Dict[str, Any]:
    """
    Generate adaptive attacks based on learned patterns
    
    Args:
        target_model: Target model name
        attack_count: Number of attacks to generate
        
    Returns:
        Generated adaptive attacks
    """
    try:
        attacks = await learning_service.generate_adaptive_attacks(target_model, attack_count)
        
        return {
            "status": "success",
            "attacks": attacks,
            "count": len(attacks),
            "target_model": target_model,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Adaptive attack generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns/{pattern_id}")
async def get_pattern_details(pattern_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific pattern
    
    Args:
        pattern_id: ID of the pattern
        
    Returns:
        Pattern details
    """
    try:
        patterns = await learning_service.get_learned_patterns()
        pattern = next((p for p in patterns if p['pattern_id'] == pattern_id), None)
        
        if not pattern:
            raise HTTPException(status_code=404, detail="Pattern not found")
        
        return {
            "status": "success",
            "pattern": pattern,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pattern details retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threats/{threat_id}")
async def get_threat_details(threat_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific threat
    
    Args:
        threat_id: ID of the threat
        
    Returns:
        Threat details
    """
    try:
        threats = await learning_service.get_threat_intelligence()
        threat = next((t for t in threats if t['threat_id'] == threat_id), None)
        
        if not threat:
            raise HTTPException(status_code=404, detail="Threat not found")
        
        return {
            "status": "success",
            "threat": threat,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Threat details retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/anomalies/{anomaly_id}")
async def get_anomaly_details(anomaly_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific anomaly
    
    Args:
        anomaly_id: ID of the anomaly
        
    Returns:
        Anomaly details
    """
    try:
        anomalies = await learning_service.get_anomalies()
        anomaly = next((a for a in anomalies if a['anomaly_id'] == anomaly_id), None)
        
        if not anomaly:
            raise HTTPException(status_code=404, detail="Anomaly not found")
        
        return {
            "status": "success",
            "anomaly": anomaly,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Anomaly details retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/clusters")
async def get_pattern_clusters() -> Dict[str, Any]:
    """
    Get pattern clusters information
    
    Returns:
        Pattern clusters
    """
    try:
        clusters = learning_service.pattern_clusters
        
        cluster_info = []
        for cluster_id, pattern_ids in clusters.items():
            cluster_info.append({
                "cluster_id": cluster_id,
                "pattern_count": len(pattern_ids),
                "pattern_ids": pattern_ids
            })
        
        return {
            "status": "success",
            "clusters": cluster_info,
            "total_clusters": len(clusters),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Pattern clusters retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/behavioral-profiles")
async def get_behavioral_profiles() -> Dict[str, Any]:
    """
    Get behavioral profiles information
    
    Returns:
        Behavioral profiles
    """
    try:
        profiles = learning_service.behavioral_profiles
        
        profile_info = []
        for source, profile in profiles.items():
            profile_info.append({
                "source": source,
                "attack_count": profile['attack_count'],
                "success_rate": profile['success_rate'],
                "attack_types": dict(profile['attack_types']),
                "threat_levels": dict(profile['threat_levels']),
                "first_seen": profile['first_seen'].isoformat(),
                "last_seen": profile['last_seen'].isoformat()
            })
        
        return {
            "status": "success",
            "profiles": profile_info,
            "total_profiles": len(profiles),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Behavioral profiles retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capabilities")
async def get_learning_capabilities() -> Dict[str, Any]:
    """
    Get continuous learning capabilities
    
    Returns:
        Learning capabilities
    """
    try:
        capabilities = {
            "learning_types": [
                {
                    "type": "pattern_learning",
                    "description": "Learn and adapt attack patterns from historical data",
                    "features": ["Pattern extraction", "Feature analysis", "Success rate tracking"]
                },
                {
                    "type": "threat_intelligence",
                    "description": "Identify and track emerging threats",
                    "features": ["Threat detection", "Indicator analysis", "Mitigation strategies"]
                },
                {
                    "type": "anomaly_detection",
                    "description": "Detect unusual patterns and behaviors",
                    "features": ["Statistical analysis", "Behavioral profiling", "Anomaly scoring"]
                },
                {
                    "type": "adaptive_attacks",
                    "description": "Generate adaptive attacks based on learned patterns",
                    "features": ["Pattern evolution", "Attack generation", "Model targeting"]
                },
                {
                    "type": "behavioral_analysis",
                    "description": "Analyze and profile attack behaviors",
                    "features": ["Behavioral modeling", "Profile tracking", "Trend analysis"]
                }
            ],
            "algorithms": [
                "TF-IDF Vectorization",
                "DBSCAN Clustering",
                "K-Means Clustering",
                "Cosine Similarity",
                "Statistical Analysis",
                "Pattern Recognition"
            ],
            "data_sources": [
                "Attack logs",
                "Security events",
                "Model predictions",
                "Threat feeds",
                "Behavioral data"
            ],
            "outputs": [
                "Learned patterns",
                "Threat intelligence",
                "Anomaly alerts",
                "Adaptive attacks",
                "Behavioral profiles",
                "Learning insights"
            ]
        }
        
        return {
            "status": "success",
            "capabilities": capabilities,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Capabilities retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset-learning")
async def reset_learning_data() -> Dict[str, Any]:
    """
    Reset all learning data (use with caution)
    
    Returns:
        Reset status
    """
    try:
        # Reset all learning data
        learning_service.attack_patterns.clear()
        learning_service.threat_intelligence.clear()
        learning_service.anomalies.clear()
        learning_service.pattern_clusters.clear()
        learning_service.behavioral_profiles.clear()
        
        return {
            "status": "success",
            "message": "All learning data has been reset",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Learning data reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def get_learning_health() -> Dict[str, Any]:
    """
    Get continuous learning service health
    
    Returns:
        Service health status
    """
    try:
        health_status = {
            "status": "healthy",
            "service": "continuous_learning",
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "total_patterns": len(learning_service.attack_patterns),
                "total_threats": len(learning_service.threat_intelligence),
                "total_anomalies": len(learning_service.anomalies),
                "pattern_clusters": len(learning_service.pattern_clusters),
                "behavioral_profiles": len(learning_service.behavioral_profiles)
            },
            "learning_models": {
                "text_vectorizer": "initialized",
                "pattern_clusterer": "initialized",
                "anomaly_detector": "initialized"
            }
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Learning health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "continuous_learning",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
