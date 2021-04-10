from typing import Optional
from ..basic_typing import Length, ShapeX


class SpatialInfo:
    """
    Represent spatial information of a n-dimensional volume
    """
    def __init__(
            self,
            origin: Optional[Length] = None,
            spacing: Optional[Length] = None,
            shape: Optional[ShapeX] = None,
        ):
        """
        Args:
            origin: a n-dimensional vector representing the distance between world origin (0, 0, 0) to the origin
                (top left voxel) of a volume defined in DHW order
            spacing: the size in mm of a voxel in each dimension, in DHW order
            shape: the shape of the volume in DHW order
        """
        self.origin: Length = origin
        self.spacing: Length = spacing
        self.shape: ShapeX = shape
