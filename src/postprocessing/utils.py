import torch
import numpy as np
import pandas as pd
import datetime
import os
from config.loc_config import MODEL_SAVE_LOC, REPORT_SAVE_LOC


def calculate_model_performance(y_true, y_pred, class_names):
    num_classes = len(set(y_true + y_pred))
    # build confusion matrix based on predictions and class_index
    confusion_matrix = torch.zeros(num_classes, num_classes)
    for i in range(len(y_pred)):
        # true label on row, predicted on column
        confusion_matrix[y_true[i], y_pred[i]] += 1

    # PER-CLASS METRICS:
    # calculate accuracy, precision, recall, f1 for each class:
    accuracy = torch.zeros(num_classes)
    precision = torch.zeros(num_classes)
    recall = torch.zeros(num_classes)
    f1_score = torch.zeros(num_classes)
    for i in range(num_classes):
        # find TP, FP, FN, TN for each class:
        TP = confusion_matrix[i, i]
        FP = torch.sum(confusion_matrix[i, :]) - TP
        FN = torch.sum(confusion_matrix[:, i]) - TP
        TN = torch.sum(confusion_matrix) - TP - FP - FN
        # calculate accuracy, precision, recall, f1 for each class:
        accuracy[i] = (TP+TN)/(TP+FP+FN+TN)
        precision[i] = TP/(TP+FP)
        recall[i] = TP/(TP+FN)
        f1_score[i] = 2*precision[i]*recall[i]/(precision[i]+recall[i])
    # calculate support for each class
    support = torch.sum(confusion_matrix, dim=0)
    # calculate support proportion for each class
    support_prop = support/torch.sum(support)

    # OVERALL METRICS
    # calculate overall accuracy:
    overall_acc = torch.sum(torch.diag(confusion_matrix)
                            )/torch.sum(confusion_matrix)
    # calculate macro average F1 score:
    macro_avg_f1_score = torch.sum(f1_score)/num_classes
    # calculate weighted average rF1 score based on support proportion:
    weighted_avg_f1_score = torch.sum(f1_score*support_prop)

    TP = torch.diag(confusion_matrix)
    FP = torch.sum(confusion_matrix, dim=1) - TP
    FN = torch.sum(confusion_matrix, dim=0) - TP
    TN = torch.sum(confusion_matrix) - (TP + FP + FN)

    # calculate micro average f1 score based on TP, FP, FN
    micro_avg_f1_score = torch.sum(
        2*TP)/(torch.sum(2*TP)+torch.sum(FP)+torch.sum(FN))

    # METRICS PRESENTATION
    # performance for each class
    class_columns = ['accuracy', 'precision', 'recall', 'f1_score']
    class_data_raw = [accuracy.numpy(), precision.numpy(),
                      recall.numpy(), f1_score.numpy()]
    class_data = np.around(class_data_raw, decimals=3)
    df_class_raw = pd.DataFrame(
        class_data, index=class_columns, columns=class_names)
    class_metrics = df_class_raw.T

    # overall performance
    overall_columns = ['accuracy', 'f1_mirco', 'f1_macro', 'f1_weighted']
    overall_data_raw = [overall_acc.numpy(), micro_avg_f1_score.numpy(
    ), macro_avg_f1_score.numpy(), weighted_avg_f1_score.numpy()]
    overall_data = np.around(overall_data_raw, decimals=3)
    overall_metrics = pd.DataFrame(
        overall_data, index=overall_columns, columns=['overall'])
    return confusion_matrix, class_metrics, overall_metrics


def generate_fn_cost_matrix(confusion_matrix):
    # set all elements of cost_matrix to zeros:
    dimension = len(confusion_matrix)
    cost_matrix = torch.zeros(dimension, dimension)
    for j in range(dimension):
        for i in range(dimension):
            cost_matrix[i, j] = confusion_matrix[i, j] / \
                (torch.sum(confusion_matrix[:, j]) - confusion_matrix[j, j])
    # set diagonal back to 0
    for i in range(dimension):
        cost_matrix[i:i+1, i:i+1] = 0
    return cost_matrix


def generate_fp_cost_matrix(confusion_matrix):
    # set all elements of cost_matrix to zeros:
    dimension = len(confusion_matrix)
    cost_matrix = torch.zeros(dimension, dimension)
    for i in range(dimension):
        for j in range(dimension):
            cost_matrix[i, j] = confusion_matrix[i, j] / \
                (torch.sum(confusion_matrix[i, :]) - confusion_matrix[i, i])
    # set diagonal back to 0
    for i in range(dimension):
        cost_matrix[i:i+1, i:i+1] = 0
    return cost_matrix

def get_current_timestamp():
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def save_model_with_timestamp(model, filepath = MODEL_SAVE_LOC):
    filename = get_current_timestamp() + '_cnn_model' + '.pt'
    filepath = os.path.join(filepath, filename)
    torch.save(model.state_dict(), filepath)
    return print('Saved model to: ', filepath)


def save_csv_with_timestamp(train_result_dict, filepath = MODEL_SAVE_LOC):
    filename = get_current_timestamp() + '_training_report' + '.csv'
    filepath = os.path.join(filepath, filename)
    df = pd.DataFrame(train_result_dict)
    df.to_csv(filepath)
    return print('Saved training report to: ', filepath)
