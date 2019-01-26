class Court:
    def __init__(self):
        self.longitude = 18000
        self.width = 9000

    @staticmethod
    def is_in_court(point):
        x, y, _ = point

        if abs(x) > 9000 and (y < 0 or y > 9000):
            return True
        else:
            return False
