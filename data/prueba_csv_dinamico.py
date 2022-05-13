import pandas as pd

if __name__ == "__main__":
    test_value = "Bromodomain"
    data = pd.read_csv("test_data.csv", index_col="Unnamed: 0")
    ids = [
        "ipp_id",
        # "chembl_id",
        # "SMILES",
        "library",
        "PPI family",
        # "PPI",
    ]
    data = data[ids]
    print(data)
    t = data["PPI family"].to_list()
    data["target"] = ["Yes" if item == test_value else "No" for item in t]
    print(data)
