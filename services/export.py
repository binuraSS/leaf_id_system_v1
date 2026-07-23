# services/export.py
"""
Data export service
"""
import json
import csv
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ExportService:
    """
    Export analysis results in various formats.
    """
    
    def __init__(self):
        self.supported_formats = ['json', 'csv', 'txt']
    
    def export_json(self, data: Dict[str, Any], 
                    filepath: Optional[str] = None) -> Optional[str]:
        """
        Export data as JSON.
        
        Args:
            data: Data to export
            filepath: Output file path
        
        Returns:
            Path to exported file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"export_{timestamp}.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Exported JSON to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"JSON export error: {e}")
            return None
    
    def export_csv(self, data: List[Dict[str, Any]], 
                   filepath: Optional[str] = None) -> Optional[str]:
        """
        Export data as CSV.
        
        Args:
            data: List of dictionaries to export
            filepath: Output file path
        
        Returns:
            Path to exported file
        """
        if not data:
            logger.error("No data to export")
            return None
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"export_{timestamp}.csv"
        
        try:
            # Flatten nested dictionaries
            flattened_data = []
            for item in data:
                flat_item = self._flatten_dict(item)
                flattened_data.append(flat_item)
            
            # Get all keys
            all_keys = set()
            for item in flattened_data:
                all_keys.update(item.keys())
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
                writer.writeheader()
                for item in flattened_data:
                    writer.writerow(item)
            
            logger.info(f"Exported CSV to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"CSV export error: {e}")
            return None
    
    def export_txt(self, data: Dict[str, Any], 
                   filepath: Optional[str] = None) -> Optional[str]:
        """
        Export data as text.
        
        Args:
            data: Data to export
            filepath: Output file path
        
        Returns:
            Path to exported file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"export_{timestamp}.txt"
        
        try:
            with open(filepath, 'w') as f:
                f.write(self._dict_to_text(data))
            logger.info(f"Exported TXT to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"TXT export error: {e}")
            return None
    
    def _flatten_dict(self, data: Dict[str, Any], 
                      parent_key: str = '', 
                      sep: str = '_') -> Dict[str, Any]:
        """
        Flatten nested dictionary.
        
        Args:
            data: Dictionary to flatten
            parent_key: Parent key for nesting
            sep: Separator for nested keys
        
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _dict_to_text(self, data: Dict[str, Any], 
                      indent: int = 0) -> str:
        """
        Convert dictionary to formatted text.
        
        Args:
            data: Dictionary to convert
            indent: Indentation level
        
        Returns:
            Formatted text string
        """
        lines = []
        prefix = "  " * indent
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._dict_to_text(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(self._dict_to_text(item, indent + 1))
                    else:
                        lines.append(f"{prefix}  - {item}")
            else:
                lines.append(f"{prefix}{key}: {value}")
        
        return "\n".join(lines)