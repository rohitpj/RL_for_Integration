import json
import csv

def to_space_joined(xs):
    # join list of mixed types safely
    return " ".join(map(str, xs))

with open("Data/BWD_train.json", "r") as f:
    data = json.load(f)

with open("Data/BWD_train.csv", "w", newline="") as f:
    writer = csv.writer(f)

    # unified header (target_tokens may be blank for new schema)
    writer.writerow(["function", "input_tokens", "target_expr", "target_tokens", "labels"])

    for entry in data:
        if len(entry) == 5:
            input_expr = entry[0]
            input_tokens = to_space_joined(entry[1])
            target_expr = entry[2]
            target_tokens = to_space_joined(entry[3])
            labels = to_space_joined(entry[4])

        elif len(entry) == 4:
            input_expr = entry[0]
            input_tokens = to_space_joined(entry[1])
            target_expr = entry[2]
            target_tokens = ""              # not provided in new format
            labels = to_space_joined(entry[3])

        else:
            raise ValueError(f"Unexpected entry length {len(entry)}: {entry[:2]} ...")

        writer.writerow([input_expr, input_tokens, target_expr, target_tokens, labels])
