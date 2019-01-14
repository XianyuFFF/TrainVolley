from Camera import Camera
from Player import Player
from Ball import Ball
from reconstruct import find_fundamental_matrix


class World:
    def __init__(self, cam0, cam1):
        super(World, self).__init__()
        self.cams = [cam0, cam1]
        self.fundamental_matrix = find_fundamental_matrix(*self.cams)
