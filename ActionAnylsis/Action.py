class Action:
    def __init__(self):
        self.start_frame = 0
        self.end_frame = 0
        self.action_name = ""

    def __str__(self):
        for key, value in vars(self):
            return "{} : {}".format(key, value)


def merge_action_sequence(action_sequence):
    # No idea now
    actions = action_sequence
    return actions