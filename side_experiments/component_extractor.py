from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class Component:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]      # (x, y, width, height) in crop coordinates
    origin_offset: tuple[int, int]        # (x, y) position in page coordinates
    area: int

class ComponentExtractor:
    def __init__(self, min_component_area: int):
        if min_component_area <= 0:
            raise ValueError("min_component_area must be positive")
        self.min_component_area = min_component_area

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
        def filter(self, components: list[Component]) -> list[Component]:
            return [c for c in components if c.area >= self.min_component_area]
