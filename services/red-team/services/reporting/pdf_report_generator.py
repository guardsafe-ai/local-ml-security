"""
PDF Report Generator for Enterprise Reporting
Generates comprehensive PDF reports for executives and auditors
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.platypus import Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics import renderPDF
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from io import BytesIO
import base64

logger = logging.getLogger(__name__)


class PDFReportGenerator:
    """
    Generates comprehensive PDF reports for enterprise stakeholders
    """
    
    def __init__(self):
        """Initialize PDF report generator"""
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        logger.info("✅ PDF Report Generator initialized")
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        try:
            # Title style
            self.styles.add(ParagraphStyle(
                name='CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.darkblue
            ))
            
            # Subtitle style
            self.styles.add(ParagraphStyle(
                name='CustomSubtitle',
                parent=self.styles['Heading2'],
                fontSize=16,
                spaceAfter=20,
                textColor=colors.darkgreen
            ))
            
            # Executive Summary style
            self.styles.add(ParagraphStyle(
                name='ExecutiveSummary',
                parent=self.styles['Normal'],
                fontSize=12,
                spaceAfter=12,
                leftIndent=20,
                rightIndent=20,
                textColor=colors.black
            ))
            
            # Risk Level styles
            self.styles.add(ParagraphStyle(
                name='CriticalRisk',
                parent=self.styles['Normal'],
                fontSize=12,
                textColor=colors.red,
                backColor=colors.lightgrey
            ))
            
            self.styles.add(ParagraphStyle(
                name='HighRisk',
                parent=self.styles['Normal'],
                fontSize=12,
                textColor=colors.orange
            ))
            
            self.styles.add(ParagraphStyle(
                name='MediumRisk',
                parent=self.styles['Normal'],
                fontSize=12,
                textColor=colors.yellow
            ))
            
            self.styles.add(ParagraphStyle(
                name='LowRisk',
                parent=self.styles['Normal'],
                fontSize=12,
                textColor=colors.green
            ))
            
        except Exception as e:
            logger.error(f"Custom styles setup failed: {e}")
    
    def generate_executive_summary_report(self, 
                                        security_data: Dict[str, Any],
                                        output_path: str) -> bool:
        """
        Generate executive summary PDF report
        
        Args:
            security_data: Security assessment data
            output_path: Path to save PDF file
            
        Returns:
            True if successful
        """
        try:
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            story = []
            
            # Title page
            story.extend(self._create_title_page(security_data))
            story.append(PageBreak())
            
            # Executive summary
            story.extend(self._create_executive_summary(security_data))
            story.append(PageBreak())
            
            # Risk overview
            story.extend(self._create_risk_overview(security_data))
            story.append(PageBreak())
            
            # Attack analysis
            story.extend(self._create_attack_analysis(security_data))
            story.append(PageBreak())
            
            # Compliance status
            story.extend(self._create_compliance_status(security_data))
            story.append(PageBreak())
            
            # Recommendations
            story.extend(self._create_recommendations(security_data))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"Executive summary report generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Executive summary report generation failed: {e}")
            return False
    
    def generate_audit_report(self, 
                            audit_data: Dict[str, Any],
                            output_path: str) -> bool:
        """
        Generate detailed audit report
        
        Args:
            audit_data: Audit trail and compliance data
            output_path: Path to save PDF file
            
        Returns:
            True if successful
        """
        try:
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            story = []
            
            # Title page
            story.extend(self._create_audit_title_page(audit_data))
            story.append(PageBreak())
            
            # Audit scope and methodology
            story.extend(self._create_audit_scope(audit_data))
            story.append(PageBreak())
            
            # Detailed findings
            story.extend(self._create_detailed_findings(audit_data))
            story.append(PageBreak())
            
            # Compliance matrix
            story.extend(self._create_compliance_matrix(audit_data))
            story.append(PageBreak())
            
            # Evidence and logs
            story.extend(self._create_evidence_section(audit_data))
            story.append(PageBreak())
            
            # Remediation tracking
            story.extend(self._create_remediation_tracking(audit_data))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"Audit report generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Audit report generation failed: {e}")
            return False
    
    def generate_technical_report(self, 
                                technical_data: Dict[str, Any],
                                output_path: str) -> bool:
        """
        Generate technical detailed report
        
        Args:
            technical_data: Technical analysis data
            output_path: Path to save PDF file
            
        Returns:
            True if successful
        """
        try:
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            story = []
            
            # Technical title page
            story.extend(self._create_technical_title_page(technical_data))
            story.append(PageBreak())
            
            # Methodology
            story.extend(self._create_methodology_section(technical_data))
            story.append(PageBreak())
            
            # Attack techniques analysis
            story.extend(self._create_attack_techniques_analysis(technical_data))
            story.append(PageBreak())
            
            # Model performance analysis
            story.extend(self._create_model_performance_analysis(technical_data))
            story.append(PageBreak())
            
            # Vulnerability details
            story.extend(self._create_vulnerability_details(technical_data))
            story.append(PageBreak())
            
            # Exploitability assessment
            story.extend(self._create_exploitability_assessment(technical_data))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"Technical report generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Technical report generation failed: {e}")
            return False
    
    def _create_title_page(self, data: Dict[str, Any]) -> List:
        """Create title page content"""
        try:
            story = []
            
            # Main title
            title = Paragraph("ML Security Assessment Report", self.styles['CustomTitle'])
            story.append(title)
            story.append(Spacer(1, 20))
            
            # Subtitle
            subtitle = Paragraph("Executive Summary & Risk Analysis", self.styles['CustomSubtitle'])
            story.append(subtitle)
            story.append(Spacer(1, 30))
            
            # Report metadata
            metadata = [
                ["Report Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                ["Assessment Period:", data.get('assessment_period', 'N/A')],
                ["Models Tested:", str(data.get('models_tested', 0))],
                ["Total Attacks:", str(data.get('total_attacks', 0))],
                ["Vulnerabilities Found:", str(data.get('vulnerabilities_found', 0))],
                ["Overall Risk Level:", data.get('overall_risk_level', 'Unknown')]
            ]
            
            metadata_table = Table(metadata, colWidths=[2*inch, 3*inch])
            metadata_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ]))
            
            story.append(metadata_table)
            story.append(Spacer(1, 30))
            
            # Confidentiality notice
            confidentiality = Paragraph(
                "<b>CONFIDENTIAL</b><br/>This report contains sensitive security information and should be handled according to your organization's data classification policies.",
                self.styles['Normal']
            )
            story.append(confidentiality)
            
            return story
            
        except Exception as e:
            logger.error(f"Title page creation failed: {e}")
            return []
    
    def _create_executive_summary(self, data: Dict[str, Any]) -> List:
        """Create executive summary section"""
        try:
            story = []
            
            # Section title
            title = Paragraph("Executive Summary", self.styles['CustomSubtitle'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Key findings
            key_findings = data.get('key_findings', [])
            if key_findings:
                findings_text = "<b>Key Findings:</b><br/>"
                for i, finding in enumerate(key_findings[:5], 1):
                    findings_text += f"{i}. {finding}<br/>"
                
                findings_para = Paragraph(findings_text, self.styles['ExecutiveSummary'])
                story.append(findings_para)
                story.append(Spacer(1, 12))
            
            # Risk summary
            risk_summary = data.get('risk_summary', {})
            if risk_summary:
                risk_text = f"""
                <b>Risk Summary:</b><br/>
                • Critical Vulnerabilities: {risk_summary.get('critical', 0)}<br/>
                • High Risk Issues: {risk_summary.get('high', 0)}<br/>
                • Medium Risk Issues: {risk_summary.get('medium', 0)}<br/>
                • Low Risk Issues: {risk_summary.get('low', 0)}<br/>
                • Overall Risk Score: {risk_summary.get('overall_score', 'N/A')}/10
                """
                
                risk_para = Paragraph(risk_text, self.styles['ExecutiveSummary'])
                story.append(risk_para)
                story.append(Spacer(1, 12))
            
            # Business impact
            business_impact = data.get('business_impact', '')
            if business_impact:
                impact_text = f"<b>Business Impact:</b><br/>{business_impact}"
                impact_para = Paragraph(impact_text, self.styles['ExecutiveSummary'])
                story.append(impact_para)
                story.append(Spacer(1, 12))
            
            # Immediate actions
            immediate_actions = data.get('immediate_actions', [])
            if immediate_actions:
                actions_text = "<b>Immediate Actions Required:</b><br/>"
                for i, action in enumerate(immediate_actions[:3], 1):
                    actions_text += f"{i}. {action}<br/>"
                
                actions_para = Paragraph(actions_text, self.styles['ExecutiveSummary'])
                story.append(actions_para)
            
            return story
            
        except Exception as e:
            logger.error(f"Executive summary creation failed: {e}")
            return []
    
    def _create_risk_overview(self, data: Dict[str, Any]) -> List:
        """Create risk overview section"""
        try:
            story = []
            
            # Section title
            title = Paragraph("Risk Overview", self.styles['CustomSubtitle'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Risk distribution chart
            risk_data = data.get('risk_distribution', {})
            if risk_data:
                # Create risk distribution table
                risk_table_data = [
                    ["Risk Level", "Count", "Percentage", "Status"],
                    ["Critical", str(risk_data.get('critical', 0)), f"{risk_data.get('critical_pct', 0)}%", "🔴"],
                    ["High", str(risk_data.get('high', 0)), f"{risk_data.get('high_pct', 0)}%", "🟠"],
                    ["Medium", str(risk_data.get('medium', 0)), f"{risk_data.get('medium_pct', 0)}%", "🟡"],
                    ["Low", str(risk_data.get('low', 0)), f"{risk_data.get('low_pct', 0)}%", "🟢"]
                ]
                
                risk_table = Table(risk_table_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch])
                risk_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
                    ('BACKGROUND', (1, 1), (1, -1), colors.beige),
                    ('BACKGROUND', (2, 1), (2, -1), colors.lightblue),
                    ('BACKGROUND', (3, 1), (3, -1), colors.lightgreen),
                ]))
                
                story.append(risk_table)
                story.append(Spacer(1, 20))
            
            # Top vulnerabilities
            top_vulnerabilities = data.get('top_vulnerabilities', [])
            if top_vulnerabilities:
                vuln_text = "<b>Top Vulnerabilities:</b><br/>"
                for i, vuln in enumerate(top_vulnerabilities[:5], 1):
                    vuln_text += f"{i}. {vuln.get('name', 'Unknown')} - {vuln.get('severity', 'Unknown')}<br/>"
                
                vuln_para = Paragraph(vuln_text, self.styles['ExecutiveSummary'])
                story.append(vuln_para)
            
            return story
            
        except Exception as e:
            logger.error(f"Risk overview creation failed: {e}")
            return []
    
    def _create_attack_analysis(self, data: Dict[str, Any]) -> List:
        """Create attack analysis section"""
        try:
            story = []
            
            # Section title
            title = Paragraph("Attack Analysis", self.styles['CustomSubtitle'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Attack statistics
            attack_stats = data.get('attack_statistics', {})
            if attack_stats:
                stats_data = [
                    ["Metric", "Value"],
                    ["Total Attacks Executed", str(attack_stats.get('total_attacks', 0))],
                    ["Successful Attacks", str(attack_stats.get('successful_attacks', 0))],
                    ["Success Rate", f"{attack_stats.get('success_rate', 0):.1f}%"],
                    ["Average Response Time", f"{attack_stats.get('avg_response_time', 0):.2f}ms"],
                    ["Detection Rate", f"{attack_stats.get('detection_rate', 0):.1f}%"]
                ]
                
                stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
                stats_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 11),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
                ]))
                
                story.append(stats_table)
                story.append(Spacer(1, 20))
            
            # Attack categories breakdown
            attack_categories = data.get('attack_categories', {})
            if attack_categories:
                cat_text = "<b>Attack Categories Breakdown:</b><br/>"
                for category, count in attack_categories.items():
                    cat_text += f"• {category}: {count} attacks<br/>"
                
                cat_para = Paragraph(cat_text, self.styles['ExecutiveSummary'])
                story.append(cat_para)
            
            return story
            
        except Exception as e:
            logger.error(f"Attack analysis creation failed: {e}")
            return []
    
    def _create_compliance_status(self, data: Dict[str, Any]) -> List:
        """Create compliance status section"""
        try:
            story = []
            
            # Section title
            title = Paragraph("Compliance Status", self.styles['CustomSubtitle'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Compliance frameworks
            compliance = data.get('compliance_status', {})
            if compliance:
                compliance_data = [
                    ["Framework", "Status", "Score", "Last Assessment"],
                    ["SOC 2", compliance.get('soc2', {}).get('status', 'N/A'), 
                     f"{compliance.get('soc2', {}).get('score', 0)}%", 
                     compliance.get('soc2', {}).get('last_assessment', 'N/A')],
                    ["ISO 27001", compliance.get('iso27001', {}).get('status', 'N/A'),
                     f"{compliance.get('iso27001', {}).get('score', 0)}%",
                     compliance.get('iso27001', {}).get('last_assessment', 'N/A')],
                    ["OWASP LLM Top 10", compliance.get('owasp', {}).get('status', 'N/A'),
                     f"{compliance.get('owasp', {}).get('score', 0)}%",
                     compliance.get('owasp', {}).get('last_assessment', 'N/A')]
                ]
                
                compliance_table = Table(compliance_data, colWidths=[2*inch, 1.5*inch, 1*inch, 1.5*inch])
                compliance_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
                ]))
                
                story.append(compliance_table)
                story.append(Spacer(1, 20))
            
            # Compliance recommendations
            recommendations = data.get('compliance_recommendations', [])
            if recommendations:
                rec_text = "<b>Compliance Recommendations:</b><br/>"
                for i, rec in enumerate(recommendations[:5], 1):
                    rec_text += f"{i}. {rec}<br/>"
                
                rec_para = Paragraph(rec_text, self.styles['ExecutiveSummary'])
                story.append(rec_para)
            
            return story
            
        except Exception as e:
            logger.error(f"Compliance status creation failed: {e}")
            return []
    
    def _create_recommendations(self, data: Dict[str, Any]) -> List:
        """Create recommendations section"""
        try:
            story = []
            
            # Section title
            title = Paragraph("Recommendations", self.styles['CustomSubtitle'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Immediate actions
            immediate = data.get('immediate_recommendations', [])
            if immediate:
                imm_text = "<b>Immediate Actions (0-30 days):</b><br/>"
                for i, rec in enumerate(immediate[:5], 1):
                    imm_text += f"{i}. {rec}<br/>"
                
                imm_para = Paragraph(imm_text, self.styles['ExecutiveSummary'])
                story.append(imm_para)
                story.append(Spacer(1, 12))
            
            # Short-term recommendations
            short_term = data.get('short_term_recommendations', [])
            if short_term:
                st_text = "<b>Short-term Actions (1-3 months):</b><br/>"
                for i, rec in enumerate(short_term[:5], 1):
                    st_text += f"{i}. {rec}<br/>"
                
                st_para = Paragraph(st_text, self.styles['ExecutiveSummary'])
                story.append(st_para)
                story.append(Spacer(1, 12))
            
            # Long-term recommendations
            long_term = data.get('long_term_recommendations', [])
            if long_term:
                lt_text = "<b>Long-term Actions (3-12 months):</b><br/>"
                for i, rec in enumerate(long_term[:5], 1):
                    lt_text += f"{i}. {rec}<br/>"
                
                lt_para = Paragraph(lt_text, self.styles['ExecutiveSummary'])
                story.append(lt_para)
            
            return story
            
        except Exception as e:
            logger.error(f"Recommendations creation failed: {e}")
            return []
    
    def _create_audit_title_page(self, data: Dict[str, Any]) -> List:
        """Create audit report title page"""
        try:
            story = []
            
            # Main title
            title = Paragraph("ML Security Audit Report", self.styles['CustomTitle'])
            story.append(title)
            story.append(Spacer(1, 20))
            
            # Audit metadata
            audit_metadata = [
                ["Audit Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                ["Audit Period:", data.get('audit_period', 'N/A')],
                ["Auditor:", data.get('auditor', 'ML Security Platform')],
                ["Audit Scope:", data.get('audit_scope', 'Full ML Security Assessment')],
                ["Compliance Frameworks:", data.get('compliance_frameworks', 'SOC 2, ISO 27001')],
                ["Total Controls Tested:", str(data.get('controls_tested', 0))],
                ["Controls Passed:", str(data.get('controls_passed', 0))],
                ["Controls Failed:", str(data.get('controls_failed', 0))],
                ["Overall Compliance Score:", f"{data.get('compliance_score', 0)}%"]
            ]
            
            audit_table = Table(audit_metadata, colWidths=[2*inch, 3*inch])
            audit_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ]))
            
            story.append(audit_table)
            story.append(Spacer(1, 30))
            
            # Audit disclaimer
            disclaimer = Paragraph(
                "<b>AUDIT DISCLAIMER</b><br/>This audit report is based on automated security testing and analysis. Manual verification may be required for critical findings.",
                self.styles['Normal']
            )
            story.append(disclaimer)
            
            return story
            
        except Exception as e:
            logger.error(f"Audit title page creation failed: {e}")
            return []
    
    def _create_audit_scope(self, data: Dict[str, Any]) -> List:
        """Create audit scope section"""
        try:
            story = []
            
            # Section title
            title = Paragraph("Audit Scope & Methodology", self.styles['CustomSubtitle'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Scope details
            scope_text = f"""
            <b>Audit Scope:</b><br/>
            {data.get('scope_description', 'Comprehensive ML security assessment covering all attack vectors and compliance requirements.')}<br/><br/>
            
            <b>Methodology:</b><br/>
            • Automated security testing using advanced ML techniques<br/>
            • OWASP LLM Top 10 compliance verification<br/>
            • CVSS v3.1 vulnerability scoring<br/>
            • Comprehensive audit logging and evidence collection<br/>
            • Real-time threat intelligence integration<br/><br/>
            
            <b>Systems Tested:</b><br/>
            • ML Models: {data.get('models_tested', 0)} models<br/>
            • Attack Patterns: {data.get('attack_patterns', 0)} patterns<br/>
            • Security Controls: {data.get('security_controls', 0)} controls<br/>
            • Compliance Frameworks: {data.get('compliance_frameworks', 'SOC 2, ISO 27001')}
            """
            
            scope_para = Paragraph(scope_text, self.styles['ExecutiveSummary'])
            story.append(scope_para)
            
            return story
            
        except Exception as e:
            logger.error(f"Audit scope creation failed: {e}")
            return []
    
    def _create_detailed_findings(self, data: Dict[str, Any]) -> List:
        """Create detailed findings section"""
        try:
            story = []
            
            # Section title
            title = Paragraph("Detailed Findings", self.styles['CustomSubtitle'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Findings table
            findings = data.get('detailed_findings', [])
            if findings:
                findings_data = [["ID", "Severity", "Category", "Description", "Status"]]
                
                for finding in findings[:20]:  # Limit to 20 findings
                    findings_data.append([
                        finding.get('id', 'N/A'),
                        finding.get('severity', 'N/A'),
                        finding.get('category', 'N/A'),
                        finding.get('description', 'N/A')[:50] + '...',
                        finding.get('status', 'N/A')
                    ])
                
                findings_table = Table(findings_data, colWidths=[0.8*inch, 0.8*inch, 1.2*inch, 2.5*inch, 0.8*inch])
                findings_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                
                story.append(findings_table)
            
            return story
            
        except Exception as e:
            logger.error(f"Detailed findings creation failed: {e}")
            return []
    
    def _create_compliance_matrix(self, data: Dict[str, Any]) -> List:
        """Create compliance matrix section"""
        try:
            story = []
            
            # Section title
            title = Paragraph("Compliance Matrix", self.styles['CustomSubtitle'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Compliance matrix data
            compliance_matrix = data.get('compliance_matrix', {})
            if compliance_matrix:
                matrix_data = [["Control", "SOC 2", "ISO 27001", "OWASP", "Status"]]
                
                for control, frameworks in compliance_matrix.items():
                    matrix_data.append([
                        control,
                        "✓" if frameworks.get('soc2', False) else "✗",
                        "✓" if frameworks.get('iso27001', False) else "✗",
                        "✓" if frameworks.get('owasp', False) else "✗",
                        "PASS" if all(frameworks.values()) else "FAIL"
                    ])
                
                matrix_table = Table(matrix_data, colWidths=[2*inch, 1*inch, 1*inch, 1*inch, 1*inch])
                matrix_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                
                story.append(matrix_table)
            
            return story
            
        except Exception as e:
            logger.error(f"Compliance matrix creation failed: {e}")
            return []
    
    def _create_evidence_section(self, data: Dict[str, Any]) -> List:
        """Create evidence section"""
        try:
            story = []
            
            # Section title
            title = Paragraph("Evidence & Audit Trail", self.styles['CustomSubtitle'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Evidence summary
            evidence = data.get('evidence', {})
            if evidence:
                evidence_text = f"""
                <b>Evidence Summary:</b><br/>
                • Audit Logs: {evidence.get('audit_logs', 0)} entries<br/>
                • Test Results: {evidence.get('test_results', 0)} results<br/>
                • Screenshots: {evidence.get('screenshots', 0)} images<br/>
                • Configuration Files: {evidence.get('config_files', 0)} files<br/>
                • Network Traces: {evidence.get('network_traces', 0)} traces<br/><br/>
                
                <b>Evidence Location:</b><br/>
                • MinIO Storage: {evidence.get('minio_path', 'N/A')}<br/>
                • Database Records: {evidence.get('db_records', 0)} records<br/>
                • File System: {evidence.get('file_system_path', 'N/A')}
                """
                
                evidence_para = Paragraph(evidence_text, self.styles['ExecutiveSummary'])
                story.append(evidence_para)
            
            return story
            
        except Exception as e:
            logger.error(f"Evidence section creation failed: {e}")
            return []
    
    def _create_remediation_tracking(self, data: Dict[str, Any]) -> List:
        """Create remediation tracking section"""
        try:
            story = []
            
            # Section title
            title = Paragraph("Remediation Tracking", self.styles['CustomSubtitle'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Remediation status
            remediation = data.get('remediation_tracking', {})
            if remediation:
                rem_data = [
                    ["Finding ID", "Priority", "Assigned To", "Due Date", "Status"],
                ]
                
                for item in remediation.get('items', []):
                    rem_data.append([
                        item.get('finding_id', 'N/A'),
                        item.get('priority', 'N/A'),
                        item.get('assigned_to', 'N/A'),
                        item.get('due_date', 'N/A'),
                        item.get('status', 'N/A')
                    ])
                
                rem_table = Table(rem_data, colWidths=[1*inch, 1*inch, 1.5*inch, 1*inch, 1*inch])
                rem_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                
                story.append(rem_table)
            
            return story
            
        except Exception as e:
            logger.error(f"Remediation tracking creation failed: {e}")
            return []
    
    def _create_technical_title_page(self, data: Dict[str, Any]) -> List:
        """Create technical report title page"""
        try:
            story = []
            
            # Main title
            title = Paragraph("Technical Security Analysis Report", self.styles['CustomTitle'])
            story.append(title)
            story.append(Spacer(1, 20))
            
            # Technical metadata
            tech_metadata = [
                ["Report Type:", "Technical Deep Dive Analysis"],
                ["Analysis Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                ["Analysis Period:", data.get('analysis_period', 'N/A')],
                ["Attack Techniques:", str(data.get('attack_techniques', 0))],
                ["Models Analyzed:", str(data.get('models_analyzed', 0))],
                ["Vulnerabilities Found:", str(data.get('vulnerabilities_found', 0))],
                ["Exploits Generated:", str(data.get('exploits_generated', 0))],
                ["Technical Risk Score:", f"{data.get('technical_risk_score', 0)}/10"]
            ]
            
            tech_table = Table(tech_metadata, colWidths=[2*inch, 3*inch])
            tech_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ]))
            
            story.append(tech_table)
            story.append(Spacer(1, 30))
            
            # Technical disclaimer
            disclaimer = Paragraph(
                "<b>TECHNICAL REPORT</b><br/>This report contains detailed technical analysis and may require specialized knowledge to fully understand. Consult with security experts for implementation guidance.",
                self.styles['Normal']
            )
            story.append(disclaimer)
            
            return story
            
        except Exception as e:
            logger.error(f"Technical title page creation failed: {e}")
            return []
    
    def _create_methodology_section(self, data: Dict[str, Any]) -> List:
        """Create methodology section"""
        try:
            story = []
            
            # Section title
            title = Paragraph("Analysis Methodology", self.styles['CustomSubtitle'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Methodology details
            methodology_text = """
            <b>Attack Techniques Used:</b><br/>
            • Gradient-based attacks (FGSM, PGD, C&W)<br/>
            • Word-level attacks (TextFooler, BERT-Attack, HotFlip)<br/>
            • Universal adversarial triggers<br/>
            • Meta-learning adaptation (MAML)<br/>
            • Genetic algorithm evolution<br/>
            • Reinforcement learning (PPO)<br/><br/>
            
            <b>Explainability Methods:</b><br/>
            • SHAP (SHapley Additive exPlanations)<br/>
            • LIME (Local Interpretable Model-agnostic Explanations)<br/>
            • Integrated Gradients<br/>
            • Attention visualization<br/><br/>
            
            <b>Evaluation Metrics:</b><br/>
            • Attack Success Rate (ASR)<br/>
            • Semantic preservation scores<br/>
            • Transferability analysis<br/>
            • CVSS v3.1 vulnerability scoring<br/>
            • OWASP LLM Top 10 compliance
            """
            
            methodology_para = Paragraph(methodology_text, self.styles['ExecutiveSummary'])
            story.append(methodology_para)
            
            return story
            
        except Exception as e:
            logger.error(f"Methodology section creation failed: {e}")
            return []
    
    def _create_attack_techniques_analysis(self, data: Dict[str, Any]) -> List:
        """Create attack techniques analysis section"""
        try:
            story = []
            
            # Section title
            title = Paragraph("Attack Techniques Analysis", self.styles['CustomSubtitle'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Attack techniques data
            techniques = data.get('attack_techniques', {})
            if techniques:
                tech_data = [["Technique", "Success Rate", "Avg Time", "Complexity", "Stealth"]]
                
                for technique, stats in techniques.items():
                    tech_data.append([
                        technique,
                        f"{stats.get('success_rate', 0):.1f}%",
                        f"{stats.get('avg_time', 0):.2f}s",
                        stats.get('complexity', 'N/A'),
                        stats.get('stealth', 'N/A')
                    ])
                
                tech_table = Table(tech_data, colWidths=[2*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
                tech_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                
                story.append(tech_table)
            
            return story
            
        except Exception as e:
            logger.error(f"Attack techniques analysis creation failed: {e}")
            return []
    
    def _create_model_performance_analysis(self, data: Dict[str, Any]) -> List:
        """Create model performance analysis section"""
        try:
            story = []
            
            # Section title
            title = Paragraph("Model Performance Analysis", self.styles['CustomSubtitle'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Model performance data
            performance = data.get('model_performance', {})
            if performance:
                perf_data = [["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Vulnerability"]]
                
                for model, stats in performance.items():
                    perf_data.append([
                        model,
                        f"{stats.get('accuracy', 0):.3f}",
                        f"{stats.get('precision', 0):.3f}",
                        f"{stats.get('recall', 0):.3f}",
                        f"{stats.get('f1_score', 0):.3f}",
                        stats.get('vulnerability_level', 'N/A')
                    ])
                
                perf_table = Table(perf_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1.5*inch])
                perf_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                
                story.append(perf_table)
            
            return story
            
        except Exception as e:
            logger.error(f"Model performance analysis creation failed: {e}")
            return []
    
    def _create_vulnerability_details(self, data: Dict[str, Any]) -> List:
        """Create vulnerability details section"""
        try:
            story = []
            
            # Section title
            title = Paragraph("Vulnerability Details", self.styles['CustomSubtitle'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Vulnerability details
            vulnerabilities = data.get('vulnerability_details', [])
            if vulnerabilities:
                for i, vuln in enumerate(vulnerabilities[:10], 1):  # Limit to 10 vulnerabilities
                    vuln_text = f"""
                    <b>Vulnerability {i}: {vuln.get('name', 'Unknown')}</b><br/>
                    • CVSS Score: {vuln.get('cvss_score', 'N/A')}<br/>
                    • Severity: {vuln.get('severity', 'N/A')}<br/>
                    • Category: {vuln.get('category', 'N/A')}<br/>
                    • Description: {vuln.get('description', 'N/A')}<br/>
                    • Exploitability: {vuln.get('exploitability', 'N/A')}<br/>
                    • Remediation: {vuln.get('remediation', 'N/A')}<br/><br/>
                    """
                    
                    vuln_para = Paragraph(vuln_text, self.styles['ExecutiveSummary'])
                    story.append(vuln_para)
            
            return story
            
        except Exception as e:
            logger.error(f"Vulnerability details creation failed: {e}")
            return []
    
    def _create_exploitability_assessment(self, data: Dict[str, Any]) -> List:
        """Create exploitability assessment section"""
        try:
            story = []
            
            # Section title
            title = Paragraph("Exploitability Assessment", self.styles['CustomSubtitle'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Exploitability data
            exploitability = data.get('exploitability_assessment', {})
            if exploitability:
                exp_text = f"""
                <b>Overall Exploitability Score: {exploitability.get('overall_score', 'N/A')}/10</b><br/><br/>
                
                <b>Exploitability Factors:</b><br/>
                • Attack Complexity: {exploitability.get('attack_complexity', 'N/A')}<br/>
                • Privileges Required: {exploitability.get('privileges_required', 'N/A')}<br/>
                • User Interaction: {exploitability.get('user_interaction', 'N/A')}<br/>
                • Scope: {exploitability.get('scope', 'N/A')}<br/><br/>
                
                <b>Impact Assessment:</b><br/>
                • Confidentiality Impact: {exploitability.get('confidentiality_impact', 'N/A')}<br/>
                • Integrity Impact: {exploitability.get('integrity_impact', 'N/A')}<br/>
                • Availability Impact: {exploitability.get('availability_impact', 'N/A')}<br/><br/>
                
                <b>Risk Assessment:</b><br/>
                • Business Risk: {exploitability.get('business_risk', 'N/A')}<br/>
                • Technical Risk: {exploitability.get('technical_risk', 'N/A')}<br/>
                • Compliance Risk: {exploitability.get('compliance_risk', 'N/A')}
                """
                
                exp_para = Paragraph(exp_text, self.styles['ExecutiveSummary'])
                story.append(exp_para)
            
            return story
            
        except Exception as e:
            logger.error(f"Exploitability assessment creation failed: {e}")
            return []
    
    def generate_report_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of report generation capabilities
        
        Args:
            data: Report data
            
        Returns:
            Summary dictionary
        """
        try:
            summary = {
                'report_types': [
                    'Executive Summary Report',
                    'Audit Report', 
                    'Technical Report'
                ],
                'features': [
                    'PDF generation with professional formatting',
                    'Executive-level summaries',
                    'Detailed technical analysis',
                    'Compliance matrix and tracking',
                    'Risk assessment and scoring',
                    'Evidence collection and audit trails',
                    'Remediation tracking',
                    'Custom styling and branding'
                ],
                'data_sources': [
                    'Security assessment results',
                    'Attack analysis data',
                    'Compliance status',
                    'Vulnerability details',
                    'Audit logs and evidence',
                    'Model performance metrics'
                ],
                'output_formats': ['PDF'],
                'target_audiences': [
                    'C-Level Executives',
                    'Security Auditors',
                    'Compliance Officers',
                    'Technical Teams',
                    'Board Members'
                ]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Report summary generation failed: {e}")
            return {"error": str(e)}
