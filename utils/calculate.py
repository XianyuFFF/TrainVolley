import numpy as np


def vector_angle(x, y):
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))

    cos_angle = x.dot(y)/(Lx * Ly)
    angle = np.arccos(cos_angle)
    return angle*360/2/np.pi

