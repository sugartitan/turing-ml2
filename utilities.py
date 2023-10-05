from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import typing
from typing import Sequence
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt


def binary_to_labeled(value: int) -> str:
    '''Changes binary features to "Yes" (1) and "No" (0)'''
    if value == 0:
        return 'No'
    
    return 'Yes'


def multiple_countplots(df: pd.DataFrame, columns: Sequence[str], target_feature: str):
    '''
    Given a pandas DataFrame, list of columns labels and target feature 
    plots countlpots for all given columns with hue set as target feature.
    '''
    fig, ax = plt.subplots(1 * ((len(columns)-1) // 3 + 1), 3, figsize=(12, 4 * ((len(columns)-1) // 3 + 1)))

    axs = ax.flatten()

    for i, col in enumerate(columns):
        counts = df.groupby([col, target_feature]).size()

        levels = [counts.index.levels[0], counts.index.levels[1]]
        new_index = pd.MultiIndex.from_product(levels, names=counts.index.names)

        counts = counts.reindex(new_index, fill_value=0)

        values = df[col].unique()
        bottom = [counts[value, 0] for value in values]
        top = [counts[value, 1] for value in values]

        axs[i].bar(values, bottom)
        axs[i].bar(values, top, bottom=bottom)
        
        for container in axs[i].containers:
            labels = [value.get_height() if value.get_height() > 0 else '' for value in container]
        
            axs[i].bar_label(container, labels=labels, label_type='center')

        for label in axs[i].get_xticklabels():
            label.set_rotation(30)

        axs[i].set_xlabel(col)

    plt.tight_layout()


def categorize_age(age: float) -> str:
    '''
    Categorizes age value into one of these categories:
    - child
    - adult
    - elder
    '''
    if age < 18:
        return 'child'
    elif age < 65:
        return 'adult'
    
    return 'elder'


def categorize_bmi(bmi: float) -> str:
    '''
    Categorizes BMI value into one of these categories:
    - underweight
    - normal
    - overweight
    - obese
    - extremely obese
    '''
    if bmi < 18.5:
        return 'underweight'
    elif bmi < 25:
        return 'normal'
    elif bmi < 30:
        return 'overweight'
    elif bmi < 35:
        return 'obese'
    
    return 'extremely obese'


def categorize_glucose(avg_glucose_level: float) -> str:
    '''
    Categorizes glucose value into one of these categories:
    - low
    - normal
    - prediabetes
    - diabetes
    '''
    if avg_glucose_level < 70:
        return 'low'
    elif avg_glucose_level < 100:
        return 'normal'
    elif avg_glucose_level < 125:
        return 'prediabetes'
    
    return 'diabetes'


def xy_split(df: pd.DataFrame, target_feature: str) -> tuple[pd.DataFrame]:
    '''
    Cuts off a target column of the DataFrame
    
    Returns two DataFrames:
    train_X and train_y
    '''
    return df.drop(columns=target_feature), df[target_feature]


def get_model_name(variable: object) -> str:
    '''Returns model name given a model'''
    return str(variable).split('(')[0]


def brute_force_models(train_X: pd.DataFrame, train_y: pd.Series, categorical_columns: Sequence[str], numerical_columns: Sequence[str], scalers: Sequence[object], encoders: Sequence[object], models: Sequence[object]) -> pd.DataFrame:
    '''
    This function tries every given combination of encoders, scalers and models.

    It needs to be provided with:
    train_X - training dataset
    train_y - target dataset
    categorical_columns - list of labels of categorical columns in train_X
    numerical_columns - list of labels of numerical columns in train_X
    scalers - list of scalers
    encoders - list of encoder
    models - list of models

    Returns a pandas DataFrame with all combinations and their accuracy, recall, precision and f1 score
    '''
    df = pd.DataFrame(columns=['encoder', 'scaler', 'model', 'accuracy', 'recall', 'precision', 'f1_score'])
    
    for encoder in encoders:
        for scaler in scalers:
            transformer = ColumnTransformer([('categorical', encoder, categorical_columns), ('numerical', scaler, numerical_columns)])
            for model in models:
                pipe = make_pipeline(transformer, model)
                
                pipe.fit(train_X, train_y)

                scores = cross_validate(pipe, train_X, train_y, scoring=['accuracy', 'recall', 'precision', 'f1'], cv=StratifiedKFold(shuffle=True, n_splits=5, random_state=42))

                accuracy = np.mean(scores['test_accuracy'])
                recall = np.mean(scores['test_recall'])
                precision = np.mean(scores['test_precision'])
                f1 = np.mean(scores['test_f1'])

                list = [get_model_name(encoder), get_model_name(scaler), get_model_name(model), accuracy, recall, precision, f1]

                df.loc[len(df)] = list

    return df


def adjust_class(pred: Sequence[float], t: float) -> list[int]:
    '''Given a list of probabilities and a threshold, returns a list with binary classes'''
    return [1 if y >= t else 0 for y in pred]


def model_report(y_test: Sequence[float], y_pred: Sequence[float]):
    '''Given y_test and y_pred provides accuracy score, confusiom matrix and classification report'''
    print('Accuracy:\n\t', accuracy_score(y_test, y_pred), '\n')
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred), '\n')
    print('Classification report:\n', classification_report(y_test, y_pred), '\n')


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    '''
    Plots precision recall curve
    '''
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", linewidth=2, label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", linewidth=2, label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')


def plot_roc_curve(fpr, tpr, label):
    '''
    Plots ROC curve given False Positive Rate and True Positive Rate
    '''
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)

    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')

    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    plt.show()