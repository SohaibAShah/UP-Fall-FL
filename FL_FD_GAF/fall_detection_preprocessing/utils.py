import numpy as np

def re_value(arr):
    """Normalize an array to the range [0, 255] and add a singleton dimension.
    
    Args:
        arr (np.ndarray): Input array to normalize.
    
    Returns:
        np.ndarray: Normalized array with shape (1, height, width).
    """
    rzero = np.min(arr)
    arr = arr + np.abs(rzero)
    r255 = np.amax(arr)
    if r255 != 0:
        fcon = 255 / r255
        arr = arr * fcon
        arr = np.expand_dims(arr, axis=0)
    return arr