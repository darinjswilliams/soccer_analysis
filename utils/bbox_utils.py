def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)

def get_center_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2)/2)


def get_bbox_width(bbox):
    return bbox[2] - bbox[0]

def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def measure_xy_distance(p1, p2):
    """
    Calculates the distance between two points in a 2D space.

    Parameters:
        p1 (tuple): A tuple representing the coordinates of the first point (x1, y1).
        p2 (tuple): A tuple representing the coordinates of the second point (x2, y2).

    Returns:
        tuple: A tuple containing the differences in the x and y coordinates (dx, dy),
               where dx is the difference in the x-coordinates and dy is the difference in the y-coordinates.
    """
    return  (p1[0] - p2[0], p1[1] - p2[1])