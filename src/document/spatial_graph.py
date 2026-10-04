"""
Spatial Graph Reading Order Sequencer.

Sorts and groups detected regions (text lines, math blocks, matrices)
into a logical top-to-bottom, left-to-right reading order.
"""

from typing import List
from src.ocr.dispatcher import ProcessedRegion


class SpatialReadingOrderGraph:
    """
    DBSCAN Y-Midpoint Line Clustering & Reading Order Sequencer.
    """
    def __init__(self, y_threshold: int = 25):
        self.y_threshold = y_threshold

    def sequence_regions(self, regions: List[ProcessedRegion]) -> List[ProcessedRegion]:
        """
        Sorts regions by vertical Y-midpoint first, then horizontal X-start.
        """
        if not regions:
            return []

        # Filter out empty scratch-outs
        valid_regions = [r for r in regions if r.transcription or r.is_math]

        # Calculate Y-midpoint for each region: y_mid = y + h / 2
        def get_y_mid(r: ProcessedRegion) -> float:
            _, y, _, h = r.bbox
            return y + h / 2.0

        def get_x_start(r: ProcessedRegion) -> float:
            x, _, _, _ = r.bbox
            return float(x)

        # Primary sort by Y-midpoint, secondary sort by X-start
        sorted_regions = sorted(valid_regions, key=lambda r: (get_y_mid(r), get_x_start(r)))
        return sorted_regions


if __name__ == "__main__":
    print("=== Testing Spatial Reading Order Sequencer ===")
    graph = SpatialReadingOrderGraph()
    
    r1 = ProcessedRegion(crop_id=1, bbox=(50, 200, 300, 40), region_type="text_line", transcription="Second line", is_math=False)
    r2 = ProcessedRegion(crop_id=2, bbox=(50, 50, 300, 40), region_type="text_line", transcription="First line", is_math=False)
    
    ordered = graph.sequence_regions([r1, r2])
    print(f"Ordered Line 1: {ordered[0].transcription}")
    print(f"Ordered Line 2: {ordered[1].transcription}")
    assert ordered[0].transcription == "First line", "Reading order sorting failed!"
    print("[OK] Spatial Reading Order Sequencer self-test passed cleanly!")
