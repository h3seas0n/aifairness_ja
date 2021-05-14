import eel
import wx
import pandas as pd
import sys
import numpy as np

# aif360 libraries:
from aif360.datasets import StandardDataset, BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import LFR, Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing, PrejudiceRemover
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, EqOddsPostprocessing, RejectOptionClassification
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing import DisparateImpactRemover

# bias mitigation
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss


def calculate_bias1(data_label, data, selection, privileged, unprivileged, fav_out):
    dataset_used = data_label
    message1_1 = "yeet1-1"
    message1_2 = "yeet1-2"
    message2_1 = "yeet2-1"
    message2_2 = "yeet2-2"
    privileged_groups = []

    try:
        fav_out = int(fav_out)
    except Exception:
        pass

    print("1. Computing.....")
    data_orig = StandardDataset(data, label_name=data_label, favorable_classes=[fav_out], # --> favorable output for biased class output
                 protected_attribute_names=[selection],
                 privileged_classes=[])

    print("Priviledged:", privileged)
    # --------------------------- output -----------------------------------
    if privileged != []:
        print("2. Computing.....")
        metric_orig_train = BinaryLabelDatasetMetric(data_orig, 
                                             unprivileged_groups=[{selection: unprivileged}],
                                             privileged_groups=[{selection: privileged}])
        print("#### Original training dataset")
        print("Statistical Parity Difference = %f" % metric_orig_train.mean_difference())
        print("Disparate Impact = %f" % metric_orig_train.disparate_impact())
        message1_1 = metric_orig_train.mean_difference()
        message1_2 = metric_orig_train.disparate_impact()

        RW = Reweighing(unprivileged_groups=[{selection: unprivileged}],
                    privileged_groups=[{selection: privileged}])
        dataset_transf_train = RW.fit_transform(data_orig)


        metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
                                                unprivileged_groups=[{selection: unprivileged}],
                                                privileged_groups=[{selection: privileged}])
        print("#### Transformed training dataset")
        print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())
        print("Disparate Impact = %f" % metric_transf_train.disparate_impact())
        message2_1 = metric_transf_train.mean_difference()
        message2_2 = metric_transf_train.disparate_impact()

    return round(message1_1, 10), round(message1_2, 10), round(message2_1, 10), round(message2_2, 10) # replace with correct return value


def calculate_bias2(data, selection, privileged, unprivileged, metrics, predicted_outcome, true_outcome):
    predicted_outcome_column_process = data.loc[:,predicted_outcome]
    predicted_outcome_column_arr = predicted_outcome_column_process.values                


    data_orig_true = BinaryLabelDataset(df = data, label_names = [true_outcome], protected_attribute_names = [selection])
    data_orig_predicted = data_orig_true.copy()
    data_orig_predicted.labels = predicted_outcome_column_arr

    privileged_group = [{selection: privileged}]
    unprivileged_group = [{selection: unprivileged}]
    metric_orig_train = ClassificationMetric(data_orig_true, data_orig_predicted, 
                                                unprivileged_groups=unprivileged_group,
                                                privileged_groups=privileged_group)

    output = [[],[]]
    for metric in metrics:
        o = eval('metric_orig_train.' + metric + '()')
        output[0].append(round(o, 10))


    RW = Reweighing(unprivileged_groups=unprivileged_group, privileged_groups=privileged_group)
    dataset_transf_train = RW.fit_transform(data_orig_predicted)

    true_outcome_column_process = data.loc[:,true_outcome]
    true_outcome_column_arr = true_outcome_column_process.values                

    data_orig_predicted_transf = dataset_transf_train
    data_orig_true_transf = data_orig_predicted_transf.copy()
    data_orig_true_transf.labels = true_outcome_column_arr
    metric_orig_train = ClassificationMetric(data_orig_true_transf, data_orig_predicted_transf, 
                                             unprivileged_groups=unprivileged_group,
                                             privileged_groups=privileged_group)
    for metric in metrics:
        o = eval('metric_orig_train.' + metric + '()')
        output[1].append(round(o, 10))

    return output


def bias_mitigation(data, target, path, pri_out, unpri_out):
    rcParams['figure.figsize'] = 14, 8
    RANDOM_SEED = 42
    LABELS = [1, 0]

    try: 
        pri_out = int(pri_out)
        unpri_out = int(unpri_out)
    except Exception:
        pass

    #Create independent and Dependent Features
    columns = data.columns.tolist()

    columns = [c for c in columns if c not in [target]]
    state = np.random.RandomState(42)
    X = data[columns]
    Y = data[target]

    # Get the Fraud and the normal dataset 
    NegetiveOutcome = data[data[target]==unpri_out]
    PositiveOutcome = data[data[target]==pri_out]

    # Implementing Oversampling for Handling Imbalanced 
    smk = SMOTETomek(random_state=42)
    X_res,y_res=smk.fit_sample(X,Y)

    data.pop('output')
    transformed_Dataset = pd.DataFrame(X_res, columns = [data.columns])
    TransCSV = transformed_Dataset.to_excel(path + "/export_dataframe.xlsx", index = False, header=True)

    return "Bias successfully mitigated at " + path

def read_data(path):
    #Input File Path #na_values parameter is hardcoded
    data = pd.read_csv(path, na_values=['Unknown', ' '])
    #data = pd.read_csv(r"C:\Users\Shizhe\Desktop\archive\heart.csv", na_values=['Unknown', ' '])
    return data

# -------------------------------- eel implementation ---------------------------------------------------------
file_path = ""

@eel.expose
def pythonPath(wildcard="*"):
    app = wx.App(None)
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    dialog = wx.FileDialog(None, 'Open', wildcard=wildcard, style=style)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
    dialog.Destroy()
    list_data = []
    data = read_data(path)
    for i in data.columns:
        list_data.append(i) 

    global file_path
    file_path = path

    return path, list_data


@eel.expose
def getInput1(data_label, selection, privileged, unprivileged, fav_out):
    output = None
    if file_path!="":
        data = read_data(file_path)
        output = calculate_bias1(data_label, data, selection, privileged, unprivileged, fav_out)
    print(output)
    
    return output


@eel.expose
def getInput2(selection, privileged, unprivileged, metrics, predicted_outcome, true_outcome):
    output = None
    print("Calculating metrics")
    if file_path!="":
        data = read_data(file_path)
    output = calculate_bias2(data, selection, privileged, unprivileged, metrics, predicted_outcome, true_outcome)
    
    return output


@eel.expose
def getInput3(outcome_column, target_path, pri_out, unpri_out):
    output = None
    print("Calculating metrics")
    if file_path!="":
        data = read_data(file_path)
    output = bias_mitigation(data, outcome_column, target_path, pri_out, unpri_out)
    
    return output
    

web_options = {
	"mode": "edge",    # change to own browser
	"host": "localhost",
	"port": 8000,
}


if __name__ == "__main__":
    eel.init('web')
    print("Initiating...")
    eel.start('index.html') #local application
    #eel.start('index.html', options=web_options, suppress_error=True) #run on web
