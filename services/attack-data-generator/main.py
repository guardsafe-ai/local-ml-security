"""
Attack Data Generator Service
Microservice for generating synthetic attack patterns and adversarial examples
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from models.requests import (
    PatternGenerationRequest,
    EvolutionaryGenerationRequest,
    ThreatIntelRequest,
    IncidentLearningRequest
)
from models.responses import (
    PatternGenerationResponse,
    EvolutionaryGenerationResponse,
    ThreatIntelResponse,
    IncidentLearningResponse,
    HealthResponse
)
from services.pattern_generator import PatternGenerator
from services.evolutionary_generator import EvolutionaryGenerator
from services.llm_based_generator import LLMBasedGenerator
from services.template_expander import TemplateExpander
from services.threat_intel_scraper import ThreatIntelScraper
from services.incident_learner import IncidentLearner
from database.connection import DatabaseManager
from database.repositories import AttackPatternRepository

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global services
pattern_generator = None
evolutionary_generator = None
llm_generator = None
template_expander = None
threat_intel_scraper = None
incident_learner = None
db_manager = None
pattern_repository = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global pattern_generator, evolutionary_generator, llm_generator
    global template_expander, threat_intel_scraper, incident_learner
    global db_manager, pattern_repository
    
    try:
        logger.info("🚀 Starting Attack Data Generator Service")
        
        # Initialize database
        db_manager = DatabaseManager()
        await db_manager.initialize()
        pattern_repository = AttackPatternRepository(db_manager)
        
        # Initialize services
        pattern_generator = PatternGenerator()
        evolutionary_generator = EvolutionaryGenerator(pattern_repository)
        llm_generator = LLMBasedGenerator()
        template_expander = TemplateExpander()
        threat_intel_scraper = ThreatIntelScraper()
        incident_learner = IncidentLearner(pattern_repository)
        
        logger.info("✅ Attack Data Generator Service initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Attack Data Generator Service: {e}")
        raise
    finally:
        # Cleanup
        if db_manager:
            await db_manager.close()
        logger.info("🛑 Attack Data Generator Service shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Attack Data Generator Service",
    description="Microservice for generating synthetic attack patterns and adversarial examples",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        return HealthResponse(
            status="healthy",
            service="attack-data-generator",
            version="1.0.0",
            timestamp=asyncio.get_event_loop().time()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Service unhealthy")


@app.post("/generate/patterns", response_model=PatternGenerationResponse)
async def generate_patterns(request: PatternGenerationRequest):
    """Generate attack patterns using various methods"""
    try:
        if not pattern_generator:
            raise HTTPException(status_code=500, detail="Pattern generator not initialized")
        
        result = await pattern_generator.generate_patterns(
            pattern_type=request.pattern_type,
            count=request.count,
            complexity=request.complexity,
            target_model=request.target_model,
            attack_category=request.attack_category
        )
        
        return PatternGenerationResponse(
            success=True,
            patterns=result["patterns"],
            generation_method=result["method"],
            quality_score=result["quality_score"],
            metadata=result["metadata"]
        )
        
    except Exception as e:
        logger.error(f"Pattern generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/evolutionary", response_model=EvolutionaryGenerationResponse)
async def generate_evolutionary_patterns(request: EvolutionaryGenerationRequest):
    """Generate attack patterns using evolutionary algorithms"""
    try:
        if not evolutionary_generator:
            raise HTTPException(status_code=500, detail="Evolutionary generator not initialized")
        
        result = await evolutionary_generator.evolve_patterns(
            initial_population=request.initial_population,
            generations=request.generations,
            population_size=request.population_size,
            mutation_rate=request.mutation_rate,
            crossover_rate=request.crossover_rate,
            fitness_function=request.fitness_function,
            target_models=request.target_models
        )
        
        return EvolutionaryGenerationResponse(
            success=True,
            evolved_patterns=result["evolved_patterns"],
            fitness_scores=result["fitness_scores"],
            generation_history=result["generation_history"],
            best_pattern=result["best_pattern"],
            metadata=result["metadata"]
        )
        
    except Exception as e:
        logger.error(f"Evolutionary generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/llm", response_model=PatternGenerationResponse)
async def generate_llm_patterns(request: PatternGenerationRequest):
    """Generate attack patterns using LLM-based generation"""
    try:
        if not llm_generator:
            raise HTTPException(status_code=500, detail="LLM generator not initialized")
        
        result = await llm_generator.generate_patterns(
            prompt=request.prompt,
            pattern_type=request.pattern_type,
            count=request.count,
            creativity_level=request.creativity_level,
            target_model=request.target_model
        )
        
        return PatternGenerationResponse(
            success=True,
            patterns=result["patterns"],
            generation_method="llm_based",
            quality_score=result["quality_score"],
            metadata=result["metadata"]
        )
        
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/template", response_model=PatternGenerationResponse)
async def generate_template_patterns(request: PatternGenerationRequest):
    """Generate attack patterns using template expansion"""
    try:
        if not template_expander:
            raise HTTPException(status_code=500, detail="Template expander not initialized")
        
        result = await template_expander.expand_templates(
            template_type=request.pattern_type,
            count=request.count,
            variables=request.variables,
            target_model=request.target_model
        )
        
        return PatternGenerationResponse(
            success=True,
            patterns=result["patterns"],
            generation_method="template_expansion",
            quality_score=result["quality_score"],
            metadata=result["metadata"]
        )
        
    except Exception as e:
        logger.error(f"Template generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/threat-intel/scrape", response_model=ThreatIntelResponse)
async def scrape_threat_intelligence(request: ThreatIntelRequest):
    """Scrape threat intelligence from various sources"""
    try:
        if not threat_intel_scraper:
            raise HTTPException(status_code=500, detail="Threat intel scraper not initialized")
        
        result = await threat_intel_scraper.scrape_intelligence(
            sources=request.sources,
            attack_types=request.attack_types,
            time_range=request.time_range,
            keywords=request.keywords
        )
        
        return ThreatIntelResponse(
            success=True,
            patterns=result["patterns"],
            sources=result["sources"],
            total_found=result["total_found"],
            metadata=result["metadata"]
        )
        
    except Exception as e:
        logger.error(f"Threat intel scraping failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/learn/incidents", response_model=IncidentLearningResponse)
async def learn_from_incidents(request: IncidentLearningRequest):
    """Learn attack patterns from production incidents"""
    try:
        if not incident_learner:
            raise HTTPException(status_code=500, detail="Incident learner not initialized")
        
        result = await incident_learner.learn_from_incidents(
            incident_data=request.incident_data,
            learning_method=request.learning_method,
            pattern_extraction=request.pattern_extraction,
            deduplication=request.deduplication
        )
        
        return IncidentLearningResponse(
            success=True,
            learned_patterns=result["learned_patterns"],
            extraction_method=result["extraction_method"],
            patterns_count=result["patterns_count"],
            metadata=result["metadata"]
        )
        
    except Exception as e:
        logger.error(f"Incident learning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patterns")
async def get_patterns(
    pattern_type: str = None,
    attack_category: str = None,
    limit: int = 100,
    offset: int = 0
):
    """Get stored attack patterns"""
    try:
        if not pattern_repository:
            raise HTTPException(status_code=500, detail="Pattern repository not initialized")
        
        patterns = await pattern_repository.get_patterns(
            pattern_type=pattern_type,
            attack_category=attack_category,
            limit=limit,
            offset=offset
        )
        
        return {
            "success": True,
            "patterns": patterns,
            "count": len(patterns),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Failed to get patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patterns/{pattern_id}")
async def get_pattern(pattern_id: str):
    """Get specific attack pattern by ID"""
    try:
        if not pattern_repository:
            raise HTTPException(status_code=500, detail="Pattern repository not initialized")
        
        pattern = await pattern_repository.get_pattern_by_id(pattern_id)
        
        if not pattern:
            raise HTTPException(status_code=404, detail="Pattern not found")
        
        return {
            "success": True,
            "pattern": pattern
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pattern {pattern_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/patterns/{pattern_id}")
async def delete_pattern(pattern_id: str):
    """Delete specific attack pattern by ID"""
    try:
        if not pattern_repository:
            raise HTTPException(status_code=500, detail="Pattern repository not initialized")
        
        success = await pattern_repository.delete_pattern(pattern_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Pattern not found")
        
        return {
            "success": True,
            "message": f"Pattern {pattern_id} deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete pattern {pattern_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_generation_stats():
    """Get generation statistics"""
    try:
        if not pattern_repository:
            raise HTTPException(status_code=500, detail="Pattern repository not initialized")
        
        stats = await pattern_repository.get_generation_stats()
        
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8010,
        reload=True,
        log_level="info"
    )
