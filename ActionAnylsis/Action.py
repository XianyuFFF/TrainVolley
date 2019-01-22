from collections import Counter



class Action:
    def __init__(self):
        self.start_frame = 0
        self.end_frame = 0
        self.action_name = ""

    def __str__(self):
        for key, value in vars(self):
            return "{} : {}".format(key, value)


def merge_action_sequence(action_sequence, window=7):
    new_action_sequence = []
    while True:
        for i, action_name in enumerate(action_sequence):
            nearby_action_names = action_sequence[max([i-window//2, 0]): min([i+window//2, len(action_sequence) - 1])]
            new_action_sequence.append(Counter(nearby_action_names).most_common(1)[0][0])
        if new_action_sequence == action_sequence:
            break
        action_sequence = new_action_sequence
    return new_action_sequence

