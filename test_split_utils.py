import pandas as pd
from split_utils import split_data


def test_group_separation():
    frame = pd.DataFrame({"lesion_id": [f"synthetic_{i//2}" for i in range(200)],
                          "label_id": [(i//2) % 2 for i in range(200)]})
    parts = split_data(frame)
    assert sum(len(part) for part in parts) == len(frame)
    groups = [set(part.lesion_id) for part in parts]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
