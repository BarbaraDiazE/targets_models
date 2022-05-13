"""Perform individual predictions"""
import os
import joblib
import json
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def get_models_names():
    json_root = "/home/babs/Documents/DIFACQUIM/targets_models/phase_a/driver_code/predictions"
    f = open(os.path.join(json_root, "models.json"))
    data_dict = json.load(f)
    return data_dict


class ECFP6:
    def __init__(self, smiles):
        self.smiles = smiles

    def get_ecfp6(self):
        ms = [Chem.MolFromSmiles(self.smiles)]
        fp = [AllChem.GetMorganFingerprintAsBitVect(x, 3) for x in ms]
        return fp

    def explicit_descriptor(self):
        fp = self.get_ecfp6()
        output = []
        for f in fp:
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(f, arr)
            output.append(arr)
        return np.asarray(output)


class PPIPrediction:
    def __init__(self, test_smiles):
        self.test_smiles = test_smiles

    @staticmethod
    def load_model(model_filename):
        root = "/home/babs/Documents/DIFACQUIM/targets_models/phase_a/results/trained_results/trained_models"
        return joblib.load(
            os.path.join(
                root, model_filename
            )
        )

    def test_data(self):
        try:
            test_data = ECFP6(self.test_smiles).explicit_descriptor()
        except:
            test_data = "Please insert a valid structure"
        return test_data

    def prediction(self, model_filename):
        model = self.load_model(model_filename)
        test_data = self.test_data()
        try:
            y = model.predict(test_data)
            prediction = str()
            if y[0] == 0:
                prediction = "Inactive"
            elif y[0] == 1:
                prediction = "Active"
            return prediction
        except:
            return "Invalid SMILES"

    def activity_against_target(self):
        model_names = get_models_names()
        print(model_names)
        filter_prediction = self.prediction(model_names["filter"])
        # print("*" * 20)
        # print("FILTER", filter_prediction)
        if filter_prediction == "Active":
            ###
            # for key, value in models_names.items():
            #     if "target" in key:
            #         print(key, self.prediction(f"ensemble_ecfp6_1_{key}.pkl"))
            ###
            target_predictions = {
                # value: [key, self.prediction(f"ensemble_ecfp6_1_{key}.pkl")]
                value: [self.prediction(f"ensemble_ecfp6_1_{key}.pkl")]
                for key, value in model_names.items()
                if "target" in key
            }
        elif filter_prediction == "Inactive":
            target_predictions = {
                value: ["Inactive"] for key, value in model_names.items() if "target" in key
            }
        else:
            target_predictions = {"": ["Insert a valid structure"]}
        target_prediction = pd.DataFrame.from_dict(target_predictions).transpose()
        target_prediction.columns = ["Prediction"]
        return target_prediction


if __name__ == "__main__":
    # smiles = "coco"
    # smiles = "Cc1ccccc1C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(O)=O"
    smiles = "CC(C)c1ccccc1Sc1ccc(cc1C(F)(F)F)-c1cc(ncn1)N1CCC[C@@H](C1)C(O)=O"

    def molecule_prediction(smiles):
        data = PPIPrediction(smiles).activity_against_target()
        return data

    data = molecule_prediction(smiles)
    print(data)
