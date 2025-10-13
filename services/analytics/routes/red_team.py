"""
Red Team Routes
"""

from fastapi import APIRouter, HTTPException
from models.requests import RedTeamTestResult
from models.responses import RedTeamSummary, ModelComparison, SuccessResponse
from services.red_team_service import RedTeamService

router = APIRouter(prefix="/red-team", tags=["red-team"])

# Initialize service
red_team_service = RedTeamService()


@router.post("/results", response_model=SuccessResponse)
async def store_red_team_results(result: RedTeamTestResult):
    """Store red team test results in database"""
    try:
        return red_team_service.store_test_result(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary", response_model=RedTeamSummary)
async def get_red_team_summary(days: int = 7):
    """Get red team test summary for the last N days"""
    try:
        return red_team_service.get_summary(days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comparison/{model_name}", response_model=ModelComparison)
async def get_model_comparison(model_name: str, days: int = 30):
    """Get comparison between pre-trained and trained versions of a model"""
    try:
        return red_team_service.get_model_comparison(model_name, days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
