# core/comparison.py
"""
Result comparison utilities for benchmarking
"""
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


def compare_results(result1: Dict[str, Any], result2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two analysis results.
    
    Args:
        result1: First result dictionary
        result2: Second result dictionary
    
    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        'timestamp': str(np.datetime64('now')),
        'metrics': {}
    }
    
    # Compare health scores
    if 'health' in result1 and 'health' in result2:
        health1 = result1['health'].overall_health_score if hasattr(result1['health'], 'overall_health_score') else 0
        health2 = result2['health'].overall_health_score if hasattr(result2['health'], 'overall_health_score') else 0
        comparison['metrics']['health_score_change'] = health2 - health1
        comparison['metrics']['health_score_direction'] = 'improved' if health2 > health1 else 'declined' if health2 < health1 else 'same'
    
    # Compare disease metrics
    if 'disease' in result1 and 'disease' in result2:
        disease1 = result1['disease']
        disease2 = result2['disease']
        comparison['metrics']['chlorosis_change'] = (disease2.chlorosis_percentage if hasattr(disease2, 'chlorosis_percentage') else 0) - \
                                                     (disease1.chlorosis_percentage if hasattr(disease1, 'chlorosis_percentage') else 0)
        comparison['metrics']['necrosis_change'] = (disease2.necrosis_percentage if hasattr(disease2, 'necrosis_percentage') else 0) - \
                                                    (disease1.necrosis_percentage if hasattr(disease1, 'necrosis_percentage') else 0)
    
    # Compare quality scores
    if 'quality' in result1 and 'quality' in result2:
        quality1 = result1['quality'].score if hasattr(result1['quality'], 'score') else 0
        quality2 = result2['quality'].score if hasattr(result2['quality'], 'score') else 0
        comparison['metrics']['quality_change'] = quality2 - quality1
    
    # Compare processing times
    if 'processing_time' in result1 and 'processing_time' in result2:
        comparison['metrics']['speed_change'] = result2['processing_time'] - result1['processing_time']
    
    # Generate summary
    comparison['summary'] = generate_comparison_summary(comparison['metrics'])
    
    return comparison


def generate_comparison_summary(metrics: Dict[str, Any]) -> str:
    """
    Generate human-readable comparison summary.
    
    Args:
        metrics: Comparison metrics dictionary
    
    Returns:
        Summary string
    """
    summary_parts = []
    
    # Health change
    if 'health_score_change' in metrics:
        change = metrics['health_score_change']
        if change > 0:
            summary_parts.append(f"Health improved by {change:.1f}%")
        elif change < 0:
            summary_parts.append(f"Health declined by {abs(change):.1f}%")
        else:
            summary_parts.append("Health unchanged")
    
    # Disease changes
    if 'chlorosis_change' in metrics:
        change = metrics['chlorosis_change']
        if change > 0:
            summary_parts.append(f"Chlorosis increased by {change:.1f}%")
        elif change < 0:
            summary_parts.append(f"Chlorosis decreased by {abs(change):.1f}%")
    
    if 'necrosis_change' in metrics:
        change = metrics['necrosis_change']
        if change > 0:
            summary_parts.append(f"Necrosis increased by {change:.1f}%")
        elif change < 0:
            summary_parts.append(f"Necrosis decreased by {abs(change):.1f}%")
    
    # Quality change
    if 'quality_change' in metrics:
        change = metrics['quality_change']
        if change > 0.05:
            summary_parts.append(f"Quality improved (score +{change:.2f})")
        elif change < -0.05:
            summary_parts.append(f"Quality declined (score {change:.2f})")
    
    # Speed change
    if 'speed_change' in metrics:
        change = metrics['speed_change']
        if change < 0:
            summary_parts.append(f"Processing faster by {abs(change):.2f}s")
        elif change > 0:
            summary_parts.append(f"Processing slower by {change:.2f}s")
    
    if not summary_parts:
        return "No significant changes detected"
    
    return " | ".join(summary_parts)


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two masks.
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
    
    Returns:
        IoU score (0-1)
    """
    if mask1 is None or mask2 is None:
        return 0.0
    
    # Ensure same size
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]))
    
    # Binarize
    mask1 = (mask1 > 127).astype(np.uint8)
    mask2 = (mask2 > 127).astype(np.uint8)
    
    # Calculate IoU
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_dice_score(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate Dice similarity coefficient between two masks.
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
    
    Returns:
        Dice score (0-1)
    """
    if mask1 is None or mask2 is None:
        return 0.0
    
    # Ensure same size
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]))
    
    # Binarize
    mask1 = (mask1 > 127).astype(np.uint8)
    mask2 = (mask2 > 127).astype(np.uint8)
    
    # Calculate Dice
    intersection = np.sum(mask1 & mask2)
    sum_masks = np.sum(mask1) + np.sum(mask2)
    
    if sum_masks == 0:
        return 0.0
    
    return 2 * intersection / sum_masks


def batch_compare(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare multiple results in batch.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Dictionary with batch comparison results
    """
    if len(results) < 2:
        return {'error': 'Need at least 2 results for comparison'}
    
    batch_comparison = {
        'num_samples': len(results),
        'pairwise_comparisons': [],
        'statistics': {}
    }
    
    # Pairwise comparisons
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            comparison = compare_results(results[i], results[j])
            comparison['sample1'] = i
            comparison['sample2'] = j
            batch_comparison['pairwise_comparisons'].append(comparison)
    
    # Calculate statistics
    health_scores = []
    quality_scores = []
    processing_times = []
    
    for result in results:
        if 'health' in result and hasattr(result['health'], 'overall_health_score'):
            health_scores.append(result['health'].overall_health_score)
        if 'quality' in result and hasattr(result['quality'], 'score'):
            quality_scores.append(result['quality'].score)
        if 'processing_time' in result:
            processing_times.append(result['processing_time'])
    
    if health_scores:
        batch_comparison['statistics']['health_mean'] = np.mean(health_scores)
        batch_comparison['statistics']['health_std'] = np.std(health_scores)
        batch_comparison['statistics']['health_min'] = np.min(health_scores)
        batch_comparison['statistics']['health_max'] = np.max(health_scores)
    
    if quality_scores:
        batch_comparison['statistics']['quality_mean'] = np.mean(quality_scores)
        batch_comparison['statistics']['quality_std'] = np.std(quality_scores)
    
    if processing_times:
        batch_comparison['statistics']['time_mean'] = np.mean(processing_times)
        batch_comparison['statistics']['time_std'] = np.std(processing_times)
    
    return batch_comparison