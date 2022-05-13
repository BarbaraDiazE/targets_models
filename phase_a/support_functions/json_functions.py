import os


def read_id_json(json_root, json_filename):
    import json

    f = open(os.path.join(json_root, json_filename))
    data_dict = json.load(f)
    return data_dict


def read_json(json_root, json_filename):
    import json
    f = open(os.path.join(json_root, json_filename))
    data_dict = json.load(f)
    return data_dict


if __name__ == "__main__":
    data = read_json(
        "/home/babs/Documents/DIFACQUIM/targets_models/phase_a/results/trained_results/id_models",
        "models_id.json"
    )
    for key, value in data.items():
        print(value)
