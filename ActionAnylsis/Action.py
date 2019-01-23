from collections import Counter, namedtuple


Action = namedtuple('Action', ['start_frame', 'end_frame', 'action_name'])


def merge_action_sequence(action_sequence, window=7):
    new_action_sequence = []
    while True:
        for i, action_name in enumerate(action_sequence):
            nearby_action_names = action_sequence[max([i-window//2, 0]): min([i+window//2, len(action_sequence) - 1])]
            new_action_sequence.append(Counter(nearby_action_names).most_common(1)[0][0])
        if new_action_sequence == action_sequence:
            break
        action_sequence = new_action_sequence

    actions = []
    pre_action_name = action_sequence[0]
    start_frame = 0
    for i, action_name in enumerate(action_sequence):
        if action_name != pre_action_name:
            actions.append(Action(start_frame=start_frame, end_frame=i-1, action_name=action_name))
            start_frame = i

    return new_action_sequence

