"""
Model evaluation functions: metrics and ROC/PR curves.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    roc_auc_score, auc as compute_auc, r2_score
)


def calculate_r2_score(y_true, y_pred):
    """
    Calculate R² score for regression.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True target values
    y_pred : np.ndarray or pd.Series
        Predicted target values
        
    Returns
    -------
    float
        R² score (between -inf and 1, higher is better)
    """
    # TODO: Implement R² calculation
    # Use sklearn's r2_score
    r2 = r2_score(y_true, y_pred)
    
    return r2



def calculate_classification_metrics(y_true, y_pred):
    """
    Calculate classification metrics.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred : np.ndarray or pd.Series
        Predicted binary labels
        
    Returns
    -------
    dict
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # TODO: Implement metrics calculation
    # Return dictionary with all four metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    return metrics


def calculate_auroc_score(y_true, y_pred_proba):
    """
    Calculate Area Under the ROC Curve (AUROC).
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
        
    Returns
    -------
    float
        AUROC score (between 0 and 1)
    """
    # TODO: Implement AUROC calculation
    # Use sklearn's roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    return roc_auc


def calculate_auprc_score(y_true, y_pred_proba):
    """
    Calculate Area Under the Precision-Recall Curve (AUPRC).
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
        
    Returns
    -------
    float
        AUPRC score (between 0 and 1)
    """
    # TODO: Implement AUPRC calculation
    # Use sklearn's average_precision_score
    prc_auc = average_precision_score(y_true, y_pred_proba)

    return prc_auc


def generate_auroc_curve(y_true, y_pred_proba, model_name="Model", 
                        output_path=None, ax=None):
    """
    Generate and plot ROC curve.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
    model_name : str
        Name of the model (for legend)
    output_path : str, optional
        Path to save figure
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
        
    Returns
    -------
    tuple
        (figure, ax) or (figure,) if ax provided
    """
    # TODO: Implement ROC curve plotting
    # - Calculate ROC curve using roc_curve()
    # - Calculate AUROC using auc()
    # - Plot curve with label showing AUROC score
    # - Add diagonal reference line
    # - Set labels: "False Positive Rate", "True Positive Rate"
    # - Save to output_path if provided
    # - Return figure and/or axes
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba) # roc curve
    roc_auc = auc(fpr, tpr) # auc
    
    # ax?
    if ax is None:
        fig, ax = plt.subplots()
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc) # roc curve
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # diagonal reference line
    plt.xlim([0.0, 1.0])  # format x axis 
    plt.ylim([0.0, 1.05]) # format y axis
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{model_name} ROC") # title with model name
    plt.legend(loc='lower right')  # legend
    
    # output path
    if output_path:
        plt.savefig(output_path)
    
    plt.show()

    return fig, ax


def generate_auprc_curve(y_true, y_pred_proba, model_name="Model",
                        output_path=None, ax=None):
    """
    Generate and plot Precision-Recall curve.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
    model_name : str
        Name of the model (for legend)
    output_path : str, optional
        Path to save figure
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
        
    Returns
    -------
    tuple
        (figure, ax) or (figure,) if ax provided
    """
    # TODO: Implement PR curve plotting
    # - Calculate precision-recall curve using precision_recall_curve()
    # - Calculate AUPRC using average_precision_score()
    # - Plot curve with label showing AUPRC score
    # - Add horizontal baseline (prevalence)
    # - Set labels: "Recall", "Precision"
    # - Save to output_path if provided
    # - Return figure and/or axes
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba) # calculate precision recall curve
    prc_auc = average_precision_score(y_true, y_pred_proba )  # calculate auprc

    if ax is None:
        fig, ax = plt.subplots()

    # plotting
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (area = %0.2f)' % prc_auc)
    plt.axhline(y=0.1, color='red', linestyle='--', label='No Skill') # horizontal baseline
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}') # title
    plt.legend(loc='lower left')
    
    # output path
    if output_path:
        plt.savefig(output_path)

    plt.show()

    return fig, ax


def plot_comparison_curves(y_true, y_pred_proba_log, y_pred_proba_knn,
                          output_path=None):
    """
    Plot ROC and PR curves for both logistic regression and k-NN side by side.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba_log : np.ndarray or pd.Series
        Predicted probabilities from logistic regression
    y_pred_proba_knn : np.ndarray or pd.Series
        Predicted probabilities from k-NN
    output_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with 2 subplots (ROC and PR curves)
    """
    # TODO: Implement comparison plotting
    # - Create figure with 1x2 subplots
    # - Left: ROC curves for both models
    # - Right: PR curves for both models
    # - Add legends with AUROC/AUPRC scores
    # - Save to output_path if provided
    # - Return figure
    # knn ROC curve and ROC 
    fpr_knn, tpr_knn, _ = roc_curve(y_true, y_pred_proba_knn)
    roc_auc_knn = auc(fpr_knn, tpr_knn)
    # knn precision recall curve and PR area
    precision_knn, recall_knn, _ = precision_recall_curve(y_true, y_pred_proba_knn)
    prc_auc_knn = average_precision_score(y_true, y_pred_proba_knn)

    # logistic regression ROC curve and ROC 
    fpr_log, tpr_log, _ = roc_curve(y_true, y_pred_proba_log)
    roc_auc_log = auc(fpr_log, tpr_log)
    # logistic regression precision recall curve and PR area
    precision_log, recall_log, _ = precision_recall_curve(y_true, y_pred_proba_log)
    prc_auc_log = average_precision_score(y_true, y_pred_proba_log)

    # create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # ax1: ROC curves for both models
    ax1.plot(fpr_knn, tpr_knn, color='darkorange', lw=2, label='ROC curve kNN (area = %0.2f)' % roc_auc_knn) # roc curve knn
    ax1.plot(fpr_log, tpr_log, color='darkviolet', lw=2, label='ROC curve logistic regression (area = %0.2f)' % roc_auc_log) # roc curve logistic regression
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # diagonal reference line
    ax1.xlim([0.0, 1.0])  # format x axis 
    ax1.ylim([0.0, 1.05]) # format y axis
    ax1.xlabel('False Positive Rate')
    ax1.ylabel('True Positive Rate')
    ax1.title("ROC Curves for kNN and Logistic Regression") # title with model name
    ax1.legend(loc='lower right')  # legend

    # ax2: PR curves for both models
    ax2.plot(recall_knn, precision_knn, color='darkorange', lw=2, label='Precision-Recall curve kNN (area = %0.2f)' % prc_auc_knn)
    ax2.plot(recall_log, precision_log, color='darkviolet', lw=2, label='Precision-Recall curve logistic regression (area = %0.2f)' % prc_auc_log)
    ax2.axhline(y=0.1, color='red', linestyle='--', label='No Skill') # horizontal baseline
    ax2.xlabel('Recall')
    ax2.ylabel('Precision')
    ax2.title(f'Precision-Recall Curves for kNN and Logistic Regression') # title
    ax2.legend(loc='lower left')

    # output path
    if output_path:
        plt.savefig(output_path)

    return fig
