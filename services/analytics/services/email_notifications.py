"""
Email Notification Service for Drift Detection
Handles sending email alerts when drift is detected
"""

import logging
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

class EmailNotificationService:
    """Service for sending email notifications for drift detection"""
    
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.sender_email = os.getenv("SENDER_EMAIL", "noreply@guardsafe-ai.com")
        self.sender_password = os.getenv("SENDER_PASSWORD", "dummy_password")
        self.recipient_emails = os.getenv("RECIPIENT_EMAILS", "admin@guardsafe-ai.com").split(",")
        self.enabled = os.getenv("EMAIL_NOTIFICATIONS_ENABLED", "true").lower() == "true"
        
        # Dummy mode - when no real credentials are provided
        self.dummy_mode = self.sender_password == "dummy_password"
        
        if self.dummy_mode:
            logger.info("📧 [EMAIL] Running in DUMMY mode - emails will be logged instead of sent")
        else:
            logger.info("📧 [EMAIL] Email notifications enabled with real SMTP credentials")
    
    def send_drift_alert(self, drift_results: Dict[str, Any], model_name: str = "unknown") -> bool:
        """
        Send email alert when drift is detected
        
        Args:
            drift_results: Results from drift detection
            model_name: Name of the model that detected drift
            
        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            # Create email content
            subject = f"🚨 ML Drift Alert - {model_name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Generate email body
            body = self._generate_drift_email_body(drift_results, model_name)
            
            # Create email message
            message = MIMEMultipart()
            message["From"] = self.sender_email
            message["To"] = ", ".join(self.recipient_emails)
            message["Subject"] = subject
            
            # Add body to email
            message.attach(MIMEText(body, "html"))
            
            if self.dummy_mode:
                # In dummy mode, just log the email
                self._log_dummy_email(subject, body, drift_results)
                return True
            else:
                # Send real email
                return self._send_real_email(message)
                
        except Exception as e:
            logger.error(f"❌ [EMAIL] Failed to send drift alert: {e}")
            return False
    
    def send_model_performance_alert(self, performance_results: Dict[str, Any], model_name: str = "unknown") -> bool:
        """
        Send email alert for model performance changes
        
        Args:
            performance_results: Results from model performance drift detection
            model_name: Name of the model
            
        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            subject = f"📊 Model Performance Alert - {model_name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Generate email body for performance changes
            body = self._generate_performance_email_body(performance_results, model_name)
            
            # Create email message
            message = MIMEMultipart()
            message["From"] = self.sender_email
            message["To"] = ", ".join(self.recipient_emails)
            message["Subject"] = subject
            
            # Add body to email
            message.attach(MIMEText(body, "html"))
            
            if self.dummy_mode:
                # In dummy mode, just log the email
                self._log_dummy_email(subject, body, performance_results)
                return True
            else:
                # Send real email
                return self._send_real_email(message)
                
        except Exception as e:
            logger.error(f"❌ [EMAIL] Failed to send performance alert: {e}")
            return False
    
    def _generate_drift_email_body(self, drift_results: Dict[str, Any], model_name: str) -> str:
        """Generate HTML email body for drift detection alerts"""
        
        # Extract key information
        total_features = drift_results.get("total_features", 0)
        drifted_features = drift_results.get("drifted_features", [])
        drift_summary = drift_results.get("drift_summary", {})
        statistical_tests = drift_results.get("statistical_tests", {})
        
        # Determine severity
        severe_count = len(drift_summary.get("severe_drift_features", []))
        moderate_count = len(drift_summary.get("moderate_drift_features", []))
        minor_count = len(drift_summary.get("minor_drift_features", []))
        
        if severe_count > 0:
            severity_color = "#dc3545"  # Red
            severity_text = "SEVERE"
        elif moderate_count > 0:
            severity_color = "#fd7e14"  # Orange
            severity_text = "MODERATE"
        elif minor_count > 0:
            severity_color = "#ffc107"  # Yellow
            severity_text = "MINOR"
        else:
            severity_color = "#28a745"  # Green
            severity_text = "NONE"
        
        # Generate feature details table
        feature_details = ""
        for feature, test_result in statistical_tests.items():
            is_drifted = test_result.get("is_drifted", False)
            drift_severity = test_result.get("drift_severity", "none")
            psi_value = test_result.get("psi_value", 0)
            ks_pvalue = test_result.get("ks_pvalue", 1)
            
            feature_details += f"""
            <tr>
                <td><strong>{feature}</strong></td>
                <td style="color: {'red' if is_drifted else 'green'}">{'YES' if is_drifted else 'NO'}</td>
                <td style="color: {severity_color}">{drift_severity.upper()}</td>
                <td>{psi_value:.3f}</td>
                <td>{ks_pvalue:.2e}</td>
            </tr>
            """
        
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: {severity_color}; color: white; padding: 20px; border-radius: 5px; }}
                .content {{ margin: 20px 0; }}
                .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .alert {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 15px 0; }}
                .footer {{ margin-top: 30px; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚨 ML Drift Detection Alert</h1>
                <p><strong>Model:</strong> {model_name}</p>
                <p><strong>Severity:</strong> <span style="background-color: white; color: {severity_color}; padding: 5px 10px; border-radius: 3px;">{severity_text}</span></p>
                <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
            
            <div class="content">
                <h2>📊 Drift Summary</h2>
                <div class="summary">
                    <p><strong>Total Features Analyzed:</strong> {total_features}</p>
                    <p><strong>Features with Drift:</strong> {len(drifted_features)} ({drift_summary.get('drift_percentage', 0):.1f}%)</p>
                    <p><strong>Severe Drift:</strong> {severe_count} features</p>
                    <p><strong>Moderate Drift:</strong> {moderate_count} features</p>
                    <p><strong>Minor Drift:</strong> {minor_count} features</p>
                </div>
                
                <h2>🔍 Feature Analysis</h2>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Drift Detected</th>
                        <th>Severity</th>
                        <th>PSI Value</th>
                        <th>KS P-value</th>
                    </tr>
                    {feature_details}
                </table>
                
                <div class="alert">
                    <h3>⚠️ Action Required</h3>
                    <p>Data drift has been detected in your ML model. This may indicate:</p>
                    <ul>
                        <li>Changes in data distribution over time</li>
                        <li>Potential model performance degradation</li>
                        <li>Need for model retraining or data pipeline updates</li>
                    </ul>
                    <p><strong>Recommended Actions:</strong></p>
                    <ul>
                        <li>Review the drifted features and their statistical significance</li>
                        <li>Investigate the root cause of the distribution changes</li>
                        <li>Consider retraining the model with recent data</li>
                        <li>Update data preprocessing pipelines if needed</li>
                    </ul>
                </div>
            </div>
            
            <div class="footer">
                <p>This alert was generated by GuardSafe AI ML Security Platform</p>
                <p>For support, contact: admin@guardsafe-ai.com</p>
            </div>
        </body>
        </html>
        """
        
        return html_body
    
    def _generate_performance_email_body(self, performance_results: Dict[str, Any], model_name: str) -> str:
        """Generate HTML email body for model performance alerts"""
        
        # Extract key information
        prediction_agreement = performance_results.get("prediction_agreement", 0)
        confidence_change = performance_results.get("confidence_change", 0)
        is_improved = performance_results.get("is_improved", False)
        recommendation = performance_results.get("recommendation", "keep_old_model")
        performance_metrics = performance_results.get("performance_metrics", {})
        
        # Determine alert type
        if is_improved:
            alert_color = "#28a745"  # Green
            alert_text = "MODEL IMPROVEMENT DETECTED"
        else:
            alert_color = "#ffc107"  # Yellow
            alert_text = "MODEL PERFORMANCE CHANGE"
        
        # Generate performance metrics table
        metrics_table = ""
        if performance_metrics:
            old_metrics = performance_metrics.get("old_model", {})
            new_metrics = performance_metrics.get("new_model", {})
            improvements = performance_metrics.get("improvements", {})
            
            metrics_table = f"""
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Old Model</th>
                    <th>New Model</th>
                    <th>Change</th>
                </tr>
                <tr>
                    <td><strong>Accuracy</strong></td>
                    <td>{old_metrics.get('accuracy', 0):.3f}</td>
                    <td>{new_metrics.get('accuracy', 0):.3f}</td>
                    <td style="color: {'green' if improvements.get('accuracy', 0) > 0 else 'red'}">{improvements.get('accuracy', 0):+.3f}</td>
                </tr>
                <tr>
                    <td><strong>F1 Score</strong></td>
                    <td>{old_metrics.get('f1_score', 0):.3f}</td>
                    <td>{new_metrics.get('f1_score', 0):.3f}</td>
                    <td style="color: {'green' if improvements.get('f1_score', 0) > 0 else 'red'}">{improvements.get('f1_score', 0):+.3f}</td>
                </tr>
                <tr>
                    <td><strong>Precision</strong></td>
                    <td>{old_metrics.get('precision', 0):.3f}</td>
                    <td>{new_metrics.get('precision', 0):.3f}</td>
                    <td style="color: {'green' if improvements.get('precision', 0) > 0 else 'red'}">{improvements.get('precision', 0):+.3f}</td>
                </tr>
                <tr>
                    <td><strong>Recall</strong></td>
                    <td>{old_metrics.get('recall', 0):.3f}</td>
                    <td>{new_metrics.get('recall', 0):.3f}</td>
                    <td style="color: {'green' if improvements.get('recall', 0) > 0 else 'red'}">{improvements.get('recall', 0):+.3f}</td>
                </tr>
            </table>
            """
        
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: {alert_color}; color: white; padding: 20px; border-radius: 5px; }}
                .content {{ margin: 20px 0; }}
                .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .alert {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 15px 0; }}
                .footer {{ margin-top: 30px; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Model Performance Alert</h1>
                <p><strong>Model:</strong> {model_name}</p>
                <p><strong>Alert Type:</strong> <span style="background-color: white; color: {alert_color}; padding: 5px 10px; border-radius: 3px;">{alert_text}</span></p>
                <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
            
            <div class="content">
                <h2>📈 Performance Summary</h2>
                <div class="summary">
                    <p><strong>Prediction Agreement:</strong> {prediction_agreement:.1%}</p>
                    <p><strong>Confidence Change:</strong> {confidence_change:+.3f}</p>
                    <p><strong>Model Improved:</strong> {'Yes' if is_improved else 'No'}</p>
                    <p><strong>Recommendation:</strong> {recommendation.replace('_', ' ').title()}</p>
                </div>
                
                {metrics_table if metrics_table else ''}
                
                <div class="alert">
                    <h3>ℹ️ Performance Analysis</h3>
                    <p>Model performance comparison has been completed. Key findings:</p>
                    <ul>
                        <li><strong>Prediction Agreement:</strong> {prediction_agreement:.1%} of predictions match between old and new models</li>
                        <li><strong>Confidence Change:</strong> Average confidence changed by {confidence_change:+.3f}</li>
                        <li><strong>Recommendation:</strong> {recommendation.replace('_', ' ').title()}</li>
                    </ul>
                </div>
            </div>
            
            <div class="footer">
                <p>This alert was generated by GuardSafe AI ML Security Platform</p>
                <p>For support, contact: admin@guardsafe-ai.com</p>
            </div>
        </body>
        </html>
        """
        
        return html_body
    
    def _log_dummy_email(self, subject: str, body: str, data: Dict[str, Any]):
        """Log email in dummy mode instead of sending"""
        logger.info("=" * 80)
        logger.info("📧 [DUMMY EMAIL] Email notification would be sent:")
        logger.info(f"📧 [DUMMY EMAIL] To: {', '.join(self.recipient_emails)}")
        logger.info(f"📧 [DUMMY EMAIL] Subject: {subject}")
        logger.info("📧 [DUMMY EMAIL] Body preview (first 200 chars):")
        logger.info(f"📧 [DUMMY EMAIL] {body[:200]}...")
        logger.info("📧 [DUMMY EMAIL] Data summary:")
        logger.info(f"📧 [DUMMY EMAIL] {json.dumps(data, indent=2, default=str)}")
        logger.info("=" * 80)
    
    def _send_real_email(self, message: MIMEMultipart) -> bool:
        """Send real email via SMTP"""
        try:
            # Create secure connection
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                
                # Send email
                text = message.as_string()
                server.sendmail(self.sender_email, self.recipient_emails, text)
                
                logger.info(f"✅ [EMAIL] Drift alert sent successfully to {', '.join(self.recipient_emails)}")
                return True
                
        except Exception as e:
            logger.error(f"❌ [EMAIL] Failed to send email: {e}")
            return False

# Global instance
email_service = EmailNotificationService()
