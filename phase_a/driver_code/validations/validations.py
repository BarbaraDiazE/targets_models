import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split, cross_val_score

from phase_a.support_functions.json_functions import read_json


class Validation:
    def __init__(self, root_data: str, data_filename: str, ppi_target: str, fraction: float):
        data = pd.read_csv(
            os.path.join(root_data, data_filename.lower()), index_col="Unnamed: 0"
        )
        self.root_data = root_data
        ids = [
            "ipp_id",
            "chembl_id",
            "SMILES",
            "library",
            "PPI family",
            "PPI",
        ]
        self.target = target
        self.fraction = fraction
        self.numerical_data = data.drop(ids, axis=1)
        ppi_family = data["PPI family"].to_list()
        data["target"] = ["Yes" if item == self.target else "No" for item in ppi_family]
        self.data = data

    def evaluate_model(self, root: str, model_filename: str, cv: int):
        """compute accuracy with kfold"""
        y = np.array(self.data["target"])
        y = label_binarize(y, classes=["No", "Yes"])
        y = np.reshape(y, int(y.shape[0]))
        x_train, x_test, y_train, y_test = train_test_split(
            self.numerical_data, y, test_size=self.fraction, random_state=1992
        )
        model = joblib.load(
            os.path.join(root, "results", "trained_results", "trained_models", model_filename)
        )
        accuracies = cross_val_score(estimator=model, X=x_train, y=y_train, cv=10)
        output_data = {
            "accuracy mean": float(accuracies.mean()),
            "accuracy std": float(accuracies.std()),
        }
        output_data = pd.DataFrame.from_dict(output_data, orient="index")
        output_filename = f'k{cv}_validation_{model_filename.replace(".pkl", ".csv")}'
        output_data.to_csv(os.path.join(root, "results", "validation_results", output_filename))
        print(f'{model_filename.replace(".pkl", "")} has been validated')


if __name__ == "__main__":
    from phase_a.support_functions.vars import local_root

    models_to_validate = [
        ("target1", "ensemble_ecfp6_1_target1.pkl"),
        ("target2", "ensemble_ecfp6_1_target2.pkl"),
        ("target3", "ensemble_ecfp6_1_target3.pkl"),
        ("target4", "ensemble_ecfp6_1_target4.pkl"),
        ("target5", "ensemble_ecfp6_1_target5.pkl"),
        ("target6", "ensemble_ecfp6_1_target6.pkl"),
        ("target7", "ensemble_ecfp6_1_target7.pkl"),
        ("target8", "ensemble_ecfp6_1_target8.pkl")
    ]
    targets_names = read_json(local_root["data"], "ppi_targets.json")
    for i in range(len(models_to_validate)):
        input_filename = "dataset_ppi_ecfp6.csv"
        model_name = models_to_validate[i][1]
        target = targets_names[models_to_validate[i][0]]
        print("target: ", target)
        A = Validation(
            root_data=local_root["data"],
            data_filename=input_filename,
            ppi_target=target,
            fraction=0.2
        )
        A.evaluate_model(root=local_root["phase_a"], model_filename=model_name, cv=20)
