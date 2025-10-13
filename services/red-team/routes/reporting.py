"""
Reporting Routes for Enterprise Reporting
Provides endpoints for PDF/Excel report generation and dashboard integration
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse
import json
import os
import tempfile
import asyncio

from services.reporting.pdf_report_generator import PDFReportGenerator
from services.reporting.excel_report_generator import ExcelReportGenerator
from services.reporting.executive_dashboard import ExecutiveDashboard, DashboardType
from services.reporting.audit_report_generator import AuditReportGenerator, ComplianceFramework

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reporting", tags=["reporting"])

# Initialize generators
pdf_generator = PDFReportGenerator()
excel_generator = ExcelReportGenerator()
executive_dashboard = ExecutiveDashboard()
audit_generator = AuditReportGenerator()


@router.post("/pdf/executive-summary")
async def generate_executive_summary_pdf(
    security_data: Dict[str, Any],
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Generate executive summary PDF report
    
    Args:
        security_data: Security assessment data
        background_tasks: Background tasks for file cleanup
        
    Returns:
        Report generation status and file path
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            output_path = tmp_file.name
        
        # Generate PDF report
        success = pdf_generator.generate_executive_summary_report(security_data, output_path)
        
        if success:
            # Schedule file cleanup after 1 hour
            background_tasks.add_task(cleanup_file, output_path, delay=3600)
            
            return {
                "status": "success",
                "message": "Executive summary PDF generated successfully",
                "file_path": output_path,
                "file_size": os.path.getsize(output_path),
                "generated_at": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "message": "Failed to generate executive summary PDF"
            }
            
    except Exception as e:
        logger.error(f"Executive summary PDF generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pdf/audit-report")
async def generate_audit_report_pdf(
    audit_data: Dict[str, Any],
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Generate audit report PDF
    
    Args:
        audit_data: Audit trail and compliance data
        background_tasks: Background tasks for file cleanup
        
    Returns:
        Report generation status and file path
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            output_path = tmp_file.name
        
        # Generate PDF report
        success = pdf_generator.generate_audit_report(audit_data, output_path)
        
        if success:
            # Schedule file cleanup after 1 hour
            background_tasks.add_task(cleanup_file, output_path, delay=3600)
            
            return {
                "status": "success",
                "message": "Audit report PDF generated successfully",
                "file_path": output_path,
                "file_size": os.path.getsize(output_path),
                "generated_at": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "message": "Failed to generate audit report PDF"
            }
            
    except Exception as e:
        logger.error(f"Audit report PDF generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pdf/technical-report")
async def generate_technical_report_pdf(
    technical_data: Dict[str, Any],
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Generate technical report PDF
    
    Args:
        technical_data: Technical analysis data
        background_tasks: Background tasks for file cleanup
        
    Returns:
        Report generation status and file path
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            output_path = tmp_file.name
        
        # Generate PDF report
        success = pdf_generator.generate_technical_report(technical_data, output_path)
        
        if success:
            # Schedule file cleanup after 1 hour
            background_tasks.add_task(cleanup_file, output_path, delay=3600)
            
            return {
                "status": "success",
                "message": "Technical report PDF generated successfully",
                "file_path": output_path,
                "file_size": os.path.getsize(output_path),
                "generated_at": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "message": "Failed to generate technical report PDF"
            }
            
    except Exception as e:
        logger.error(f"Technical report PDF generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/excel/executive-dashboard")
async def generate_executive_dashboard_excel(
    security_data: Dict[str, Any],
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Generate executive dashboard Excel workbook
    
    Args:
        security_data: Security assessment data
        background_tasks: Background tasks for file cleanup
        
    Returns:
        Report generation status and file path
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            output_path = tmp_file.name
        
        # Generate Excel workbook
        success = excel_generator.generate_executive_dashboard(security_data, output_path)
        
        if success:
            # Schedule file cleanup after 1 hour
            background_tasks.add_task(cleanup_file, output_path, delay=3600)
            
            return {
                "status": "success",
                "message": "Executive dashboard Excel workbook generated successfully",
                "file_path": output_path,
                "file_size": os.path.getsize(output_path),
                "generated_at": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "message": "Failed to generate executive dashboard Excel workbook"
            }
            
    except Exception as e:
        logger.error(f"Executive dashboard Excel generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/excel/audit-workbook")
async def generate_audit_workbook_excel(
    audit_data: Dict[str, Any],
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Generate audit workbook Excel file
    
    Args:
        audit_data: Audit trail and compliance data
        background_tasks: Background tasks for file cleanup
        
    Returns:
        Report generation status and file path
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            output_path = tmp_file.name
        
        # Generate Excel workbook
        success = excel_generator.generate_audit_workbook(audit_data, output_path)
        
        if success:
            # Schedule file cleanup after 1 hour
            background_tasks.add_task(cleanup_file, output_path, delay=3600)
            
            return {
                "status": "success",
                "message": "Audit workbook Excel file generated successfully",
                "file_path": output_path,
                "file_size": os.path.getsize(output_path),
                "generated_at": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "message": "Failed to generate audit workbook Excel file"
            }
            
    except Exception as e:
        logger.error(f"Audit workbook Excel generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/excel/technical-analysis")
async def generate_technical_analysis_excel(
    technical_data: Dict[str, Any],
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Generate technical analysis Excel workbook
    
    Args:
        technical_data: Technical analysis data
        background_tasks: Background tasks for file cleanup
        
    Returns:
        Report generation status and file path
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            output_path = tmp_file.name
        
        # Generate Excel workbook
        success = excel_generator.generate_technical_analysis_workbook(technical_data, output_path)
        
        if success:
            # Schedule file cleanup after 1 hour
            background_tasks.add_task(cleanup_file, output_path, delay=3600)
            
            return {
                "status": "success",
                "message": "Technical analysis Excel workbook generated successfully",
                "file_path": output_path,
                "file_size": os.path.getsize(output_path),
                "generated_at": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "message": "Failed to generate technical analysis Excel workbook"
            }
            
    except Exception as e:
        logger.error(f"Technical analysis Excel generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/executive")
async def get_executive_dashboard() -> Dict[str, Any]:
    """
    Get executive dashboard data
    
    Returns:
        Executive dashboard data
    """
    try:
        dashboard_data = executive_dashboard.get_dashboard_data(DashboardType.EXECUTIVE)
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Executive dashboard data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/technical")
async def get_technical_dashboard() -> Dict[str, Any]:
    """
    Get technical dashboard data
    
    Returns:
        Technical dashboard data
    """
    try:
        dashboard_data = executive_dashboard.get_dashboard_data(DashboardType.TECHNICAL)
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Technical dashboard data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/compliance")
async def get_compliance_dashboard() -> Dict[str, Any]:
    """
    Get compliance dashboard data
    
    Returns:
        Compliance dashboard data
    """
    try:
        dashboard_data = executive_dashboard.get_dashboard_data(DashboardType.COMPLIANCE)
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Compliance dashboard data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/real-time")
async def get_realtime_dashboard() -> Dict[str, Any]:
    """
    Get real-time dashboard data
    
    Returns:
        Real-time dashboard data
    """
    try:
        dashboard_data = executive_dashboard.get_dashboard_data(DashboardType.REAL_TIME)
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Real-time dashboard data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dashboard/update-metrics")
async def update_dashboard_metrics(security_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update dashboard metrics with latest security data
    
    Args:
        security_data: Latest security assessment data
        
    Returns:
        Update status
    """
    try:
        success = await executive_dashboard.update_metrics(security_data)
        
        if success:
            return {
                "status": "success",
                "message": "Dashboard metrics updated successfully",
                "updated_at": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "message": "Failed to update dashboard metrics"
            }
            
    except Exception as e:
        logger.error(f"Dashboard metrics update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/metrics-summary")
async def get_metrics_summary() -> Dict[str, Any]:
    """
    Get metrics summary
    
    Returns:
        Metrics summary data
    """
    try:
        summary = executive_dashboard.get_metrics_summary()
        return summary
        
    except Exception as e:
        logger.error(f"Metrics summary retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dashboard/clear-alerts")
async def clear_dashboard_alerts(alert_ids: List[str] = None) -> Dict[str, Any]:
    """
    Clear dashboard alerts
    
    Args:
        alert_ids: List of alert IDs to clear (if None, clears all)
        
    Returns:
        Clear status
    """
    try:
        executive_dashboard.clear_alerts(alert_ids)
        
        return {
            "status": "success",
            "message": f"Cleared {len(alert_ids) if alert_ids else 'all'} alerts",
            "cleared_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Alert clearing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dashboard/mark-alert-resolved")
async def mark_alert_resolved(alert_id: str) -> Dict[str, Any]:
    """
    Mark a specific alert as resolved
    
    Args:
        alert_id: ID of the alert to mark as resolved
        
    Returns:
        Mark status
    """
    try:
        executive_dashboard.mark_alert_resolved(alert_id)
        
        return {
            "status": "success",
            "message": f"Alert {alert_id} marked as resolved",
            "resolved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Alert resolution marking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audit/initialize")
async def initialize_audit(
    audit_scope: str,
    frameworks: List[str],
    auditor: str = "ML Security Platform"
) -> Dict[str, Any]:
    """
    Initialize audit with scope and frameworks
    
    Args:
        audit_scope: Description of audit scope
        frameworks: List of compliance frameworks to audit
        auditor: Name of auditor
        
    Returns:
        Initialization status
    """
    try:
        # Convert string frameworks to enum
        framework_enums = []
        for framework in frameworks:
            try:
                framework_enums.append(ComplianceFramework(framework.lower()))
            except ValueError:
                logger.warning(f"Unknown framework: {framework}")
        
        if not framework_enums:
            raise HTTPException(status_code=400, detail="No valid frameworks provided")
        
        success = await audit_generator.initialize_audit(audit_scope, framework_enums, auditor)
        
        if success:
            return {
                "status": "success",
                "message": "Audit initialized successfully",
                "audit_id": audit_generator.audit_metadata.get('audit_id'),
                "frameworks": [f.value for f in framework_enums],
                "initialized_at": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "message": "Failed to initialize audit"
            }
            
    except Exception as e:
        logger.error(f"Audit initialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audit/test-control")
async def test_audit_control(
    control_id: str,
    test_results: Dict[str, Any],
    evidence: List[str] = None,
    tester: str = "ML Security Platform"
) -> Dict[str, Any]:
    """
    Test a specific control and record results
    
    Args:
        control_id: ID of the control to test
        test_results: Results of the control test
        evidence: List of evidence files/records
        tester: Name of the tester
        
    Returns:
        Test status
    """
    try:
        success = await audit_generator.test_control(control_id, test_results, evidence, tester)
        
        if success:
            return {
                "status": "success",
                "message": f"Control {control_id} tested successfully",
                "tested_at": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "message": f"Failed to test control {control_id}"
            }
            
    except Exception as e:
        logger.error(f"Control testing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit/status")
async def get_audit_status() -> Dict[str, Any]:
    """
    Get current audit status
    
    Returns:
        Audit status data
    """
    try:
        status = audit_generator.get_audit_status()
        return status
        
    except Exception as e:
        logger.error(f"Audit status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audit/generate-report")
async def generate_audit_report() -> Dict[str, Any]:
    """
    Generate comprehensive audit report
    
    Returns:
        Audit report data
    """
    try:
        report = await audit_generator.generate_audit_report()
        
        return {
            "status": "success",
            "message": "Audit report generated successfully",
            "report": report,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Audit report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{file_path:path}")
async def download_file(file_path: str) -> FileResponse:
    """
    Download generated report file
    
    Args:
        file_path: Path to the file to download
        
    Returns:
        File response
    """
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Determine content type based on file extension
        if file_path.endswith('.pdf'):
            media_type = 'application/pdf'
        elif file_path.endswith('.xlsx'):
            media_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        else:
            media_type = 'application/octet-stream'
        
        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=os.path.basename(file_path)
        )
        
    except Exception as e:
        logger.error(f"File download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capabilities")
async def get_reporting_capabilities() -> Dict[str, Any]:
    """
    Get reporting capabilities summary
    
    Returns:
        Capabilities summary
    """
    try:
        capabilities = {
            "pdf_reports": {
                "executive_summary": "Executive-level security assessment reports",
                "audit_report": "Detailed audit reports for compliance",
                "technical_report": "Technical deep-dive analysis reports"
            },
            "excel_workbooks": {
                "executive_dashboard": "Multi-sheet executive dashboard workbook",
                "audit_workbook": "Comprehensive audit tracking workbook",
                "technical_analysis": "Technical analysis and metrics workbook"
            },
            "dashboards": {
                "executive": "C-level executive dashboard",
                "technical": "Technical team dashboard",
                "compliance": "Compliance officer dashboard",
                "real_time": "Real-time monitoring dashboard"
            },
            "audit_frameworks": {
                "soc2": "SOC 2 Type II compliance",
                "iso27001": "ISO 27001 compliance",
                "owasp_llm": "OWASP LLM Top 10 compliance",
                "nist": "NIST Cybersecurity Framework",
                "pci_dss": "PCI DSS compliance",
                "hipaa": "HIPAA compliance"
            },
            "features": [
                "Professional PDF generation with charts and tables",
                "Interactive Excel workbooks with conditional formatting",
                "Real-time dashboard updates",
                "Comprehensive audit trail tracking",
                "Multi-framework compliance support",
                "Automated evidence collection",
                "Remediation planning and tracking",
                "Executive summary generation",
                "Technical deep-dive analysis",
                "Compliance matrix generation"
            ]
        }
        
        return capabilities
        
    except Exception as e:
        logger.error(f"Capabilities retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def cleanup_file(file_path: str, delay: int = 3600):
    """
    Clean up temporary file after specified delay
    
    Args:
        file_path: Path to the file to clean up
        delay: Delay in seconds before cleanup
    """
    try:
        await asyncio.sleep(delay)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.error(f"File cleanup failed for {file_path}: {e}")
