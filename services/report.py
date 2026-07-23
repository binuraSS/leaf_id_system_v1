# services/report.py
"""
Report generation service
"""
import json
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ReportService:
    """
    Generate reports from analysis results.
    """
    
    def __init__(self):
        self.report_template = {
            'title': 'Leaf Analysis Report',
            'generated': None,
            'summary': {},
            'metrics': {},
            'images': {},
            'recommendations': []
        }
    
    def generate_report(self, result: Dict[str, Any], 
                       include_images: bool = False) -> Dict[str, Any]:
        """
        Generate a report from analysis results.
        
        Args:
            result: Analysis result dictionary
            include_images: Whether to include base64 images
        
        Returns:
            Report dictionary
        """
        report = self.report_template.copy()
        report['generated'] = datetime.now().isoformat()
        
        # Extract summary
        if 'health' in result:
            report['summary']['health_score'] = result['health'].get('score', 0)
            report['summary']['nitrogen_status'] = result['health'].get('nitrogen_status', 'Unknown')
            report['summary']['stress_level'] = result['health'].get('stress_level', 'Unknown')
        
        if 'disease' in result:
            report['summary']['disease_status'] = result['disease'].get('status', 'Healthy')
            report['summary']['chlorosis'] = result['disease'].get('chlorosis', 0)
            report['summary']['necrosis'] = result['disease'].get('necrosis', 0)
        
        if 'quality' in result:
            report['summary']['quality_grade'] = result['quality'].get('grade', 'F')
        
        # Extract metrics
        report['metrics'] = {
            'morphology': result.get('morphology', {}),
            'color': result.get('color_metrics', {}),
            'disease_details': result.get('disease', {}),
            'processing_time': result.get('processing_time', 0)
        }
        
        # Add recommendations
        if 'disease' in result and 'recommendations' in result['disease']:
            report['recommendations'] = result['disease']['recommendations']
        
        return report
    
    def save_report(self, report: Dict[str, Any], 
                   filepath: str = None) -> str:
        """
        Save report to JSON file.
        
        Args:
            report: Report dictionary
            filepath: Path to save report
        
        Returns:
            Path to saved file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"report_{timestamp}.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            return None
    
    def to_markdown(self, report: Dict[str, Any]) -> str:
        """
        Convert report to Markdown format.
        
        Args:
            report: Report dictionary
        
        Returns:
            Markdown string
        """
        lines = [
            f"# {report.get('title', 'Leaf Analysis Report')}",
            "",
            f"**Generated:** {report.get('generated', 'Unknown')}",
            "",
            "## Summary",
            ""
        ]
        
        # Add summary
        for key, value in report.get('summary', {}).items():
            lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
        
        lines.extend(["", "## Metrics", ""])
        
        # Add metrics
        for key, value in report.get('metrics', {}).items():
            if isinstance(value, dict):
                lines.append(f"### {key.replace('_', ' ').title()}")
                for k, v in value.items():
                    lines.append(f"- **{k.replace('_', ' ').title()}:** {v}")
                lines.append("")
        
        # Add recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            lines.extend(["## Recommendations", ""])
            for rec in recommendations:
                lines.append(f"- {rec}")
        
        return "\n".join(lines)