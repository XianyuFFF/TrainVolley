class Ball:
    def __init__(self, trajectory):
        self.trajectory = trajectory

    def fastest_speed(self):
        return

    def average_spped(self):
        return

    def land_position(self):
        return

    def land_time(self):
        return

    def fly_duration_time(self):
        return

    def position_in_given_x(self, condition):
        return

    def height_over_net(self, net):
        for i, path in enumerate(self.trajectory.paths):
            if path.state == "fly":
                net_position_x = net.position.x
                return self.position_in_given_x(net_position_x).z - net.height
