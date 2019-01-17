# Need to split ball's state into "possessed", "fly" and "beat"
# if ball's movement is same as player's hand's movement, ball's state will be "possessed"
# if ball's fly change, there may be "beat" or "ground" or "net"
Ball_State = {"possessed", "fly", "beat", "ground", "net", "roll", "toss"}


class Trajectory:
    def __init__(self):
        pass