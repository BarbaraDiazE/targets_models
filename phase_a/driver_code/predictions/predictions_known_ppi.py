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


MODELS_PATH = os.getenv("MODELS_PATH")


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
        return joblib.load(os.path.join("/home/babs/Documents/DIFACQUIM/targets_models/phase_a/results/trained_results/trained_models", model_filename))
        # return joblib.load(os.path.join(f"{MODELS_PATH}", model_filename))

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
        models_names = get_models_names()
        filter_prediction = self.prediction(models_names["filter"])
        # print("*" * 20)
        # print("FILTER", filter_prediction)
        if filter_prediction == "Active":
            ###
            for key, value in models_names.items():
                if "target" in key:
                    print(key, self.prediction(f"ensemble_ecfp6_1_{key}.pkl"))
            ###
            target_predictions = {
                # value: [key, self.prediction(f"ensemble_ecfp6_1_{key}.pkl")]
                value: [self.prediction(f"ensemble_ecfp6_1_{key}.pkl")]
                for key, value in models_names.items()
                if "target" in key
            }
        elif filter_prediction == "Inactive":
            target_predictions = {
                value: ["Inactive"] for key, value in models_names.items() if "target" in key
            }
        else:
            target_predictions = {"": ["Insert a valid structure"]}
        target_prediction = pd.DataFrame.from_dict(target_predictions).transpose()
        target_prediction.columns = ["Prediction"]
        return target_prediction


if __name__ == "__main__":
    smiles = "CC1(C)CCC(CN2CCN(CC2)C2=CC=C(C(=O)NS(=O)(=O)C3=CC=C(NCC4CCOCC4)C(=C3)[N+]([O-])=O)C(OC3=CN=C4NC=CC4=C3)=C2)=C(C1)C1=CC=C(Cl)C=C1"
    model_names = get_models_names()


    def molecule_prediction(smiles):
        data = [
            PPIPrediction(smiles).prediction(f"ensemble_ecfp6_1_target{n+1}.pkl")
            for n in range(8)
        ]
        return data

    #Get ppi smiles (all population)
    from phase_a.support_functions.vars import local_root
    data = pd.read_csv(os.path.join(
        local_root["data"], "dataset_ppi_ecfp6.csv"), index_col="Unnamed: 0")
    data = data.sample(frac=0.1, random_state=1992)
    smiles_list = data["SMILES"].to_list()
    #store predictions in dict
    data_dict = {item: molecule_prediction(item) for item in smiles_list}
    #convert dict to df
    df = pd.DataFrame.from_dict(data_dict).transpose()
    c = [value for key, value in model_names.items() if "pkl" not in value]
    df.columns = c
    df["ipp_id"] = data["ipp_id"].to_list()
    df["PPI family"] = data['PPI family'].to_list()
    df.to_csv("ppi_predictions_1.csv")


