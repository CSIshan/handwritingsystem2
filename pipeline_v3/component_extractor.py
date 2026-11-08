from dataclasses import dataclass
import numpy as np
import cv2

@dataclass(eq=False)
class Component:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]      # (x, y, width, height) in crop coordinates
    origin_offset: tuple[int, int]        # (x, y) position in page coordinates
    area: int

class ComponentExtractor:
    def __init__(self, min_component_area: int, max_merge_height_px: int):
        if min_component_area <= 0:
            raise ValueError("min_component_area must be positive")
        if max_merge_height_px <= 0:
            raise ValueError("max_merge_height_px must be positive")
        self.min_component_area = min_component_area
        self.max_merge_height_px = max_merge_height_px
    def extract(
        self,
        word_crop: np.ndarray,
        page_offset: tuple[int, int],
    ) -> list[Component]:
        if word_crop.ndim != 2:
            raise ValueError("word_crop must be a 2D binary array")
        if word_crop.dtype != np.uint8:
            raise ValueError("word_crop must be dtype uint8 (0/255)")

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            word_crop, connectivity=8
        )

        components = []
        page_x, page_y = page_offset

        # label 0 is always background, skip it
        for label_id in range(1, num_labels):
            x, y, w, h, area = stats[label_id]

            mask = word_crop[y:y + h, x:x + w].copy()

            components.append(
                Component(
                    mask=mask,
                    bbox=(x, y, w, h),
                    origin_offset=(page_x + x, page_y + y),
                    area=int(area),
                )
            )

        return components
    
    def merge_nearby(self, kept: list[Component], size_ratio: float = 0.2) -> list[Component]:
        """
        kept: output of filter() -- already-filtered components.
        Merges a component into a nearby larger neighbor if it's small
        RELATIVE to that neighbor (not just "failed the noise floor").
        Single pass -- each component merges at most once.
        """
        result = list(kept)
        used = set()

        for i, small in enumerate(kept):
            if id(small) in used:
                continue
            for j, large in enumerate(kept):
                if i == j or id(large) in used:
                    continue
                if large.area <= small.area:
                    continue
                if small.area / large.area > size_ratio:
                    continue

                sx, sy, sw, sh = small.bbox
                lx, ly, lw, lh = large.bbox
                x_overlap = min(sx + sw, lx + lw) - max(sx, lx)
                if x_overlap <= 0:
                    continue

                combined_height = max(sy + sh, ly + lh) - min(sy, ly)
                if combined_height > self.max_merge_height_px:
                    continue

                merged = self._merge_pair(small, large)
                result.remove(small)
                result.remove(large)
                result.append(merged)
                used.add(id(small))
                used.add(id(large))
                break

        return result
    
    def filter(self, components: list[Component]) -> list[Component]:
        return [c for c in components if c.area >= self.min_component_area]

    def _merge_pair(self, a: Component, b: Component) -> Component:
        ax, ay, aw, ah = a.bbox
        bx, by, bw, bh = b.bbox

        x0 = min(ax, bx)
        y0 = min(ay, by)
        x1 = max(ax + aw, bx + bw)
        y1 = max(ay + ah, by + bh)
        w, h = x1 - x0, y1 - y0

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[ay - y0:ay - y0 + ah, ax - x0:ax - x0 + aw] = np.maximum(
            mask[ay - y0:ay - y0 + ah, ax - x0:ax - x0 + aw], a.mask
 )
        mask[by - y0:by - y0 + bh, bx - x0:bx - x0 + bw] = np.maximum(
            mask[by - y0:by - y0 + bh, bx - x0:bx - x0 + bw], b.mask
    )

        page_x = a.origin_offset[0] + (x0 - ax)
        page_y = a.origin_offset[1] + (y0 - ay)

        return Component(
            mask=mask,
            bbox=(x0, y0, w, h),
            origin_offset=(page_x, page_y),
            area=int(np.sum(mask > 0)),  # real ink area, not bbox area
        )
	
