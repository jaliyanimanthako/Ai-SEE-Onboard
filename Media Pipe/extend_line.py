import numpy as np
def extend_line(x1, y1, x2, y2, length=100):
    """Extend the line beyond the points (x1, y1) and (x2, y2) by a given length."""
    # Calculate the direction vector
    dx, dy = x2 - x1, y2 - y1
    # Normalize the direction vector
    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    dx, dy = dx / magnitude, dy / magnitude
    # Extend the line in both directions
    x1_extended = int(x1 - dx * length)
    y1_extended = int(y1 - dy * length)
    x2_extended = int(x2 + dx * length)
    y2_extended = int(y2 + dy * length)
    return x1_extended, y1_extended, x2_extended, y2_extended
