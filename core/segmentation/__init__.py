# core/segmentation/__init__.py
"""
Leaf segmentation module - YOLO + SAM integration
"""
from core.segmentation.isolator import (
    isolate_leaf,
    get_leaf_bounding_box,
    isolate_leaf_for_analysis,
    get_isolated_with_transparent_background,
    create_mask_overlay,
    compare_isolated_modes
)
from core.segmentation.box_selector import select_best_box, get_tight_bbox, crop_to_bbox
from core.segmentation.background import (
    try_sam_with_fallback,
    generate_sam_mask,
    otsu_fallback,
    clean_mask,
    apply_alpha_channel
)
from core.segmentation.mask_processor import (
    process_mask,
    refine_mask,
    remove_small_components,
    fill_holes,
    smooth_boundaries,
    edge_guided_refinement,
    extract_leaf_region,
    combine_masks,
    normalize_mask
)

__all__ = [
    # Isolator
    'isolate_leaf',
    'get_leaf_bounding_box',
    'isolate_leaf_for_analysis',
    'get_isolated_with_transparent_background',
    'create_mask_overlay',
    'compare_isolated_modes',
    
    # Box Selector
    'select_best_box',
    'get_tight_bbox',
    'crop_to_bbox',
    
    # Background
    'try_sam_with_fallback',
    'generate_sam_mask',
    'otsu_fallback',
    'clean_mask',
    'apply_alpha_channel',
    
    # Mask Processor
    'process_mask',
    'refine_mask',
    'remove_small_components',
    'fill_holes',
    'smooth_boundaries',
    'edge_guided_refinement',
    'extract_leaf_region',
    'combine_masks',
    'normalize_mask',
]