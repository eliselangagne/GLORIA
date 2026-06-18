# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 14:32:22 2025

@author: elise
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
from sklearn.tree import export_text, plot_tree
import json
import hashlib
from datetime import datetime
from sklearn.calibration import CalibratedClassifierCV
pd.set_option('future.no_silent_downcasting', True)


#%%%%% GLORIA V1


def GLORIA_v1(
    database_path: str,
    data_unknown_path: str,
    target_columns: list[str]
):
    """
    Train Random Forest models to predict one or more target labels 
    (e.g., Century, Region) from chemical composition data.

    This version includes:
    - Data cleaning (handling text artifacts like '<LOD', 'REF', etc.)
    - Normalization of numeric features
    - Flexible specification of one or more target columns
    - Automatic label encoding and model training per target
    - Prediction on unknown samples
    - Export of results to Excel

    Args:
        database_path (str): Path to the Excel file containing the training dataset.
        data_unknown_path (str): Path to the Excel file containing unknown glass compositions.
        output_path (str): Path to save the Excel file with predictions.
        target_columns (list[str]): List of target column names to predict (e.g. ['Century', 'Global region']).

    Returns:
        pd.DataFrame: DataFrame containing predictions for each unknown sample.
    """

    # === 1. Load data ===
    data_known = pd.read_excel(database_path)
    data_unknown = pd.read_excel(data_unknown_path)
    output_path= data_unknown_path.split('.xlsx')[0] + '_v1.xlsx'

    # === 2. Select feature columns ===
    exclude_cols = target_columns + [
        'ID', 'Reference', 'ID(Ref)', 'Site', 'City', 'Country', 'Region',
        'Global region', 'Form', 'Data Method?', 'Colour', 'Mg/Ca', 'Mg',
        'Date - Early', 'Date - Mean', 'Date - Late', 'Date',
        '(Na2O + MgO)/Sommes des autres'
    ]
    feature_columns = data_known.columns.difference(exclude_cols)
    common_columns = feature_columns.intersection(data_unknown.columns)

    X_known = data_known[common_columns].copy()
    X_unknown = data_unknown[common_columns].copy()

    # === 3. Clean text and missing values ===
    replace_map = {
        '': 0, 'REF': 0, '#VALUE!': 0, '-': 0,
        '< LOD': 0, '<LOD': 0, np.nan: 0
    }
    X_known.replace(replace_map, inplace=True)
    X_unknown.replace(replace_map, inplace=True)

    # Convert to numeric (coerce errors to NaN → replace with 0)
    X_known = X_known.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_unknown = X_unknown.apply(pd.to_numeric, errors='coerce').fillna(0)

    # === 4. Extract targets dynamically ===
    y_dict = {target: data_known[target].copy() for target in target_columns}

    # === 5. Normalize features ===
    scaler = StandardScaler()
    X_known_scaled = scaler.fit_transform(X_known)
    X_unknown_scaled = scaler.transform(X_unknown)

    # === 6. Encode labels and train models ===
    models = {}
    label_encoders = {}
    predictions = {}

    for target_name, y in y_dict.items():
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        label_encoders[target_name] = le

        # Train Random Forest
        rf = RandomForestClassifier(
            random_state=42,
            n_estimators=150,
            class_weight='balanced'
        )
        rf.fit(X_known_scaled, y_encoded)
        models[target_name] = rf

        # Predict on unknown data
        preds = rf.predict(X_unknown_scaled)
        preds_labels = le.inverse_transform(preds)
        predictions[target_name] = preds_labels

    # === 7. Export results ===
    results = pd.DataFrame(predictions)
    if 'Samples' in data_unknown.columns:
        results['Samples'] = data_unknown['Samples']
    results.to_excel(output_path, index=False)

    print(f"✅ Predictions saved to: {output_path}")
    return results


# Example usage:
predictions = GLORIA_v1(
    database_path='Database.xlsx',
    data_unknown_path='Samples_to_train.xlsx',
    target_columns=['Century', 'Global region']
)

#%%%%% GLORIA V2


def GLORIA_v2(
    database_path: str,
    data_unknown_path: str,
    target_columns: list[str],
    n_estimators: int = 150,
    random_state: int = 42,
    cv_folds: int = 5
):
    """
    Version 2:
    Train Random Forest models to predict one or more target labels 
    (e.g., Century, Region) from chemical composition data.

    New in this version:
        - Cross-validation accuracy estimation
        - Prediction confidence scores based on predict_proba()

    Args:
        database_path (str): Path to Excel file containing training data.
        data_unknown_path (str): Path to Excel file with unknown compositions.
        output_path (str): Path to save Excel file with predictions.
        target_columns (list[str]): List of target column names to predict.
        n_estimators (int): Number of trees in the Random Forest.
        random_state (int): Random seed for reproducibility.
        cv_folds (int): Number of folds for cross-validation.

    Returns:
        pd.DataFrame: DataFrame containing predictions and confidence scores.
    """

    # === 1. Load data ===
    data_known = pd.read_excel(database_path)
    data_unknown = pd.read_excel(data_unknown_path)
    output_path= data_unknown_path.split('.xlsx')[0] + '_v2.xlsx'

    # === 2. Select feature columns ===
    exclude_cols = target_columns + [
        'ID', 'Reference', 'ID(Ref)', 'Site', 'City', 'Country', 'Region',
        'Global region', 'Form', 'Data Method?', 'Colour', 'Mg/Ca', 'Mg',
        'Date - Early', 'Date - Mean', 'Date - Late', 'Date',
        '(Na2O + MgO)/Sommes des autres'
    ]
    feature_columns = data_known.columns.difference(exclude_cols)
    common_columns = feature_columns.intersection(data_unknown.columns)

    X_known = data_known[common_columns].copy()
    X_unknown = data_unknown[common_columns].copy()

    # === 3. Clean text and missing values ===
    replace_map = {
        '': 0, 'REF': 0, '#VALUE!': 0, '-': 0,
        '< LOD': 0, '<LOD': 0, np.nan: 0
    }
    X_known.replace(replace_map, inplace=True)
    X_unknown.replace(replace_map, inplace=True)

    X_known = X_known.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_unknown = X_unknown.apply(pd.to_numeric, errors='coerce').fillna(0)

    # === 4. Extract targets dynamically ===
    y_dict = {target: data_known[target].copy() for target in target_columns}

    # === 5. Normalize features ===
    scaler = StandardScaler()
    X_known_scaled = scaler.fit_transform(X_known)
    X_unknown_scaled = scaler.transform(X_unknown)

    # === 6. Encode labels, train, and evaluate models ===
    models = {}
    label_encoders = {}
    predictions = {}
    confidences = {}

    for target_name, y in y_dict.items():
        print(f"\n--- Training model for target: {target_name} ---")

        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        label_encoders[target_name] = le

        # Initialize model
        rf = RandomForestClassifier(
            random_state=random_state,
            n_estimators=n_estimators,
            class_weight='balanced'
        )

        # Cross-validation accuracy
        scores = cross_val_score(rf, X_known_scaled, y_encoded, cv=cv_folds, scoring='accuracy')
        print(f"Cross-validation accuracy for {target_name}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

        # Fit on full dataset
        rf.fit(X_known_scaled, y_encoded)
        models[target_name] = rf

        # Predict on unknown samples
        proba = rf.predict_proba(X_unknown_scaled)
        pred_idx = np.argmax(proba, axis=1)
        pred_labels = le.inverse_transform(pred_idx)
        confidence = np.max(proba, axis=1)

        predictions[target_name] = pred_labels
        confidences[f"{target_name}_confidence"] = confidence

    # === 7. Export results ===
    results = pd.DataFrame(predictions)
    for conf_col, values in confidences.items():
        results[conf_col] = values
    if 'Samples' in data_unknown.columns:
        results['Samples'] = data_unknown['Samples']

    results.to_excel(output_path, index=False)
    print(f"\n✅ Predictions and confidence scores saved to: {output_path}")

    return results


# Example usage:
predictions = GLORIA_v2(
    database_path='Database.xlsx',
    data_unknown_path='Samples_to_train.xlsx',
    target_columns=['Century', 'Global region'],
    n_estimators=200,
    cv_folds=5
)


#%%%% GLORIA V3



def GLORIA_v3(
    database_path: str,
    data_unknown_path: str,
    target_columns: list[str],
    optimize_hyperparams: bool = True,
    n_iter_search: int = 25,
    cv_folds: int = 5,
    random_state: int = 42
):
    """
    Version 3:
    Improves predictive performance by optimizing Random Forest hyperparameters.

    Includes:
    - Data cleaning and normalization
    - Cross-validation accuracy estimation
    - Prediction confidence scores
    - Optional hyperparameter optimization using RandomizedSearchCV

    Args:
        database_path (str): Path to training Excel file.
        data_unknown_path (str): Path to unknown data Excel file.
        output_path (str): Path to save predictions.
        target_columns (list[str]): List of target columns to predict.
        optimize_hyperparams (bool): Whether to perform hyperparameter search.
        n_iter_search (int): Number of random combinations tested.
        cv_folds (int): Number of folds for CV.
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Predictions with confidence scores.
    """

    # === 1. Load and prepare data ===
    data_known = pd.read_excel(database_path)
    data_unknown = pd.read_excel(data_unknown_path)
    output_path= data_unknown_path.split('.xlsx')[0] + '_v3.xlsx'

    exclude_cols = target_columns + [
        'ID', 'Reference', 'ID(Ref)', 'Site', 'City', 'Country', 'Region',
        'Global region', 'Form', 'Data Method?', 'Colour', 'Mg/Ca', 'Mg',
        'Date - Early', 'Date - Mean', 'Date - Late', 'Date',
        '(Na2O + MgO)/Sommes des autres'
    ]
    feature_columns = data_known.columns.difference(exclude_cols)
    common_columns = feature_columns.intersection(data_unknown.columns)

    X_known = data_known[common_columns].copy()
    X_unknown = data_unknown[common_columns].copy()

    replace_map = {
        '': 0, 'REF': 0, '#VALUE!': 0, '-': 0,
        '< LOD': 0, '<LOD': 0, np.nan: 0
    }
    X_known.replace(replace_map, inplace=True)
    X_unknown.replace(replace_map, inplace=True)

    X_known = X_known.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_unknown = X_unknown.apply(pd.to_numeric, errors='coerce').fillna(0)

    y_dict = {target: data_known[target].copy() for target in target_columns}

    scaler = StandardScaler()
    X_known_scaled = scaler.fit_transform(X_known)
    X_unknown_scaled = scaler.transform(X_unknown)

    models = {}
    label_encoders = {}
    predictions = {}
    confidences = {}

    # === 2. Training loop per target ===
    for target_name, y in y_dict.items():
        print(f"\n--- Optimizing model for target: {target_name} ---")

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        label_encoders[target_name] = le

        # Define base RF
        rf = RandomForestClassifier(random_state=random_state, class_weight='balanced')

        # Optional hyperparameter optimization
        if optimize_hyperparams:
            param_dist = {
                'n_estimators': np.arange(100, 501, 50),
                'max_depth': [None, 5, 10, 15, 20, 30],
                'min_samples_split': np.arange(2, 11),
                'min_samples_leaf': np.arange(1, 6),
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
            cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            search = RandomizedSearchCV(
                rf, param_distributions=param_dist,
                n_iter=n_iter_search, cv=cv_strategy,
                scoring='accuracy', random_state=random_state, n_jobs=-1
            )
            search.fit(X_known_scaled, y_encoded)
            rf = search.best_estimator_
            print(f"Best parameters for {target_name}: {search.best_params_}")
            print(f"Best CV accuracy: {search.best_score_:.3f}")
        else:
            # Simple model if optimization is disabled
            rf.set_params(n_estimators=150)
            rf.fit(X_known_scaled, y_encoded)
            cv_scores = cross_val_score(rf, X_known_scaled, y_encoded, cv=cv_folds, scoring='accuracy')
            print(f"Cross-val accuracy for {target_name}: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        # Train final model on all data
        rf.fit(X_known_scaled, y_encoded)
        models[target_name] = rf

        # Predict unknowns with probabilities
        proba = rf.predict_proba(X_unknown_scaled)
        pred_idx = np.argmax(proba, axis=1)
        pred_labels = le.inverse_transform(pred_idx)
        confidence = np.max(proba, axis=1)

        predictions[target_name] = pred_labels
        confidences[f"{target_name}_confidence"] = confidence

        # Save trained model
        joblib.dump(rf, f"{target_name}_best_model.joblib")

    # === 3. Export results ===
    results = pd.DataFrame(predictions)
    for conf_col, values in confidences.items():
        results[conf_col] = values
    if 'Samples' in data_unknown.columns:
        results['Samples'] = data_unknown['Samples']

    results.to_excel(output_path, index=False)
    print(f"\n✅ Predictions saved to: {output_path}")

    return results


# Example usage:
predictions = GLORIA_v3(
    database_path='Database.xlsx',
    data_unknown_path='Samples_to_train.xlsx',
    target_columns=['Century', 'Global region'],
    optimize_hyperparams=True,
    n_iter_search=30
)

#%%%% GLORIA V4



def GLORIA_v4(
    database_path: str,
    data_unknown_path: str,
    target_columns: list[str],
    optimize_hyperparams: bool = True,
    resample_method: str | None = None,  # Options: None, "smote", "undersample"
    n_iter_search: int = 25,
    cv_folds: int = 5,
    random_state: int = 42
):
    """
    GLORIA_v4 — Robust version (Oct. 2025)
    Handles:
    - Consistent feature scaling per target
    - Safe class balancing (SMOTE/undersample)
    - Shape verification between X and y
    - Confidence scores for predictions
    """

    # === 1. Load and clean data ===
    data_known = pd.read_excel(database_path)
    data_unknown = pd.read_excel(data_unknown_path)
    output_path = data_unknown_path.split('.xlsx')[0] + '_v4.xlsx'

    exclude_cols = target_columns + [
        'ID', 'Reference', 'ID(Ref)', 'Site', 'City', 'Country', 'Region',
        'Global region', 'Form', 'Data Method?', 'Colour', 'Mg/Ca', 'Mg',
        'Date - Early', 'Date - Mean', 'Date - Late', 'Date',
        '(Na2O + MgO)/Sommes des autres'
    ]

    # Définir les colonnes de caractéristiques communes
    feature_columns = [c for c in data_known.columns if c not in exclude_cols]
    common_columns = [c for c in feature_columns if c in data_unknown.columns]

    X_known = data_known[common_columns].copy()
    X_unknown = data_unknown[common_columns].copy()

    # Nettoyage des valeurs
    replace_map = {
        '': 0, 'REF': 0, '#VALUE!': 0, '-': 0,
        '< LOD': 0, '<LOD': 0, np.nan: 0
    }
    X_known.replace(replace_map, inplace=True)
    X_unknown.replace(replace_map, inplace=True)

    X_known = X_known.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_unknown = X_unknown.apply(pd.to_numeric, errors='coerce').fillna(0)

    models = {}
    label_encoders = {}
    predictions = {}
    confidences = {}

    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # === 2. Training loop per target ===
    for target_name in target_columns:
        print(f"\n{'='*60}")
        print(f"--- Training model for target: {target_name} ---")

        # Re-scale for each target (avoid contamination from previous iteration)
        scaler = StandardScaler()
        X_known_scaled = scaler.fit_transform(X_known)
        X_unknown_scaled = scaler.transform(X_unknown)

        # Encode target
        y = data_known[target_name].copy()
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        label_encoders[target_name] = le

        # Vérification cohérence X / y
        assert X_known_scaled.shape[0] == len(y_encoded), (
            f"Shape mismatch for {target_name}: "
            f"X_known_scaled={X_known_scaled.shape[0]}, y_encoded={len(y_encoded)}"
        )

        # Distribution initiale
        unique, counts = np.unique(y_encoded, return_counts=True)
        print("Original class distribution:")
        for cls, cnt in zip(unique, counts):
            print(f"  {le.inverse_transform([cls])[0]}: {cnt} samples")

        # === 3. Optional resampling ===
        if resample_method == "smote":
            if len(np.unique(y_encoded)) < 2:
                print(f"⚠️  Skipping SMOTE for {target_name} (only one class present).")
            else:
                print("\nApplying SMOTE oversampling...")
                smote = SMOTE(random_state=random_state)
                X_known_scaled, y_encoded = smote.fit_resample(X_known_scaled, y_encoded)
        elif resample_method == "undersample":
            print("\nApplying random undersampling...")
            rus = RandomUnderSampler(random_state=random_state)
            X_known_scaled, y_encoded = rus.fit_resample(X_known_scaled, y_encoded)

        # Distribution après rééchantillonnage
        if resample_method:
            unique, counts = np.unique(y_encoded, return_counts=True)
            print("Balanced class distribution:")
            for cls, cnt in zip(unique, counts):
                print(f"  {le.inverse_transform([cls])[0]}: {cnt} samples")

        # === 4. Define base Random Forest ===
        rf = RandomForestClassifier(random_state=random_state, class_weight='balanced')

        # === 5. Optional hyperparameter optimization ===
        if optimize_hyperparams:
            param_dist = {
                'n_estimators': np.arange(300, 501, 50),
                'max_depth': [30, 40, 50],
                'min_samples_split': np.arange(2, 11),
                'min_samples_leaf': [1],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }

            search = RandomizedSearchCV(
                rf,
                param_distributions=param_dist,
                n_iter=n_iter_search,
                cv=cv_strategy,
                scoring='accuracy',
                random_state=random_state,
                n_jobs=-1
            )

            search.fit(X_known_scaled, y_encoded)
            rf = search.best_estimator_
            print(f"\nBest parameters for {target_name}: {search.best_params_}")
            print(f"Best CV accuracy: {search.best_score_:.3f}")
        else:
            cv_scores = cross_val_score(rf, X_known_scaled, y_encoded, cv=cv_strategy, scoring='accuracy')
            print(f"CV accuracy for {target_name}: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            rf.fit(X_known_scaled, y_encoded)

        # === 6. Train final model ===
        rf.fit(X_known_scaled, y_encoded)
        models[target_name] = rf

        # === 7. Predict unknowns ===
        proba = rf.predict_proba(X_unknown_scaled)
        pred_idx = np.argmax(proba, axis=1)
        pred_labels = le.inverse_transform(pred_idx)
        confidence = np.max(proba, axis=1)

        predictions[target_name] = pred_labels
        confidences[f"{target_name}_confidence"] = confidence.round(2)

        # === 8. Save model ===
        joblib.dump(rf, f"{target_name}_balanced_model.joblib")

    # === 9. Export results ===
    results = pd.DataFrame(predictions)
    for conf_col, values in confidences.items():
        results[conf_col] = values
    if 'Samples' in data_unknown.columns:
        results['Samples'] = data_unknown['Samples']

    results.to_excel(output_path, index=False)
    print(f"\n✅ Predictions saved to: {output_path}")

    return results



# Example usage:
predictions = GLORIA_v4(
    database_path='Database.xlsx',
    data_unknown_path='Samples_to_train.xlsx',
    target_columns=['Century', 'Global region'],
    optimize_hyperparams=True,
    resample_method="smote",  # Options: None, "smote", "undersample"
    n_iter_search=30
)


#%%%% GLORIA V5


def GLORIA_v5(
    database_path: str,
    data_unknown_path: str,
    target_columns: list[str],
    optimize_hyperparams: bool = True,
    resample_method: str | None = None,  # Options: None, "smote", "undersample"
    n_iter_search: int = 25,
    cv_folds: int = 5,
    random_state: int = 42
):
    """
    GLORIA_v5 — Enhanced version (Oct. 2025)
    Adds:
    - Confusion matrices for model evaluation
    - SHAP feature importance plots
    - Automatic plot saving in Results/Plots/
    """

    # === 1. Load and clean data ===
    data_known = pd.read_excel(database_path)
    data_unknown = pd.read_excel(data_unknown_path)
    output_path = data_unknown_path.split('.xlsx')[0] + '_v5.xlsx'

    exclude_cols = target_columns + [
        'ID', 'Reference', 'ID(Ref)', 'Site', 'City', 'Country', 'Region',
        'Global region', 'Form', 'Data Method?', 'Colour', 'Mg/Ca', 'Mg',
        'Date - Early', 'Date - Mean', 'Date - Late', 'Date',
        '(Na2O + MgO)/Sommes des autres'
    ]

    feature_columns = [c for c in data_known.columns if c not in exclude_cols]
    common_columns = [c for c in feature_columns if c in data_unknown.columns]

    X_known = data_known[common_columns].copy()
    X_unknown = data_unknown[common_columns].copy()

    replace_map = {'': 0, 'REF': 0, '#VALUE!': 0, '-': 0, '< LOD': 0, '<LOD': 0, np.nan: 0}
    X_known.replace(replace_map, inplace=True)
    X_unknown.replace(replace_map, inplace=True)

    X_known = X_known.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_unknown = X_unknown.apply(pd.to_numeric, errors='coerce').fillna(0)

    models, label_encoders, predictions, confidences = {}, {}, {}, {}

    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # === Create output folders ===
    plots_dir = os.path.join("Results", "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    # === 2. Training loop per target ===
    for target_name in target_columns:
        print(f"\n{'='*60}")
        print(f"--- Training model for target: {target_name} ---")

        scaler = StandardScaler()
        X_known_scaled = scaler.fit_transform(X_known)
        X_unknown_scaled = scaler.transform(X_unknown)

        # Encode target
        y = data_known[target_name].copy()
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        label_encoders[target_name] = le

        # Vérification cohérence
        assert X_known_scaled.shape[0] == len(y_encoded), (
            f"Shape mismatch for {target_name}: X_known_scaled={X_known_scaled.shape[0]}, y_encoded={len(y_encoded)}"
        )

        # Distribution initiale
        unique, counts = np.unique(y_encoded, return_counts=True)
        print("Original class distribution:")
        for cls, cnt in zip(unique, counts):
            print(f"  {le.inverse_transform([cls])[0]}: {cnt} samples")

        # === 3. Optional resampling ===
        if resample_method == "smote":
            if len(np.unique(y_encoded)) < 2:
                print(f"⚠️  Skipping SMOTE for {target_name} (only one class present).")
            else:
                print("\nApplying SMOTE oversampling...")
                smote = SMOTE(random_state=random_state)
                X_known_scaled, y_encoded = smote.fit_resample(X_known_scaled, y_encoded)
        elif resample_method == "undersample":
            print("\nApplying random undersampling...")
            rus = RandomUnderSampler(random_state=random_state)
            X_known_scaled, y_encoded = rus.fit_resample(X_known_scaled, y_encoded)

        # Distribution après rééchantillonnage
        if resample_method:
            unique, counts = np.unique(y_encoded, return_counts=True)
            print("Balanced class distribution:")
            for cls, cnt in zip(unique, counts):
                print(f"  {le.inverse_transform([cls])[0]}: {cnt} samples")

        # === 4. Define and optimize model ===
        rf = RandomForestClassifier(random_state=random_state, class_weight='balanced')

        if optimize_hyperparams:
            param_dist = {
                'n_estimators': np.arange(300, 501, 50),
                'max_depth': [30, 40, 50],
                'min_samples_split': np.arange(2, 11),
                'min_samples_leaf': [1],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }

            search = RandomizedSearchCV(
                rf,
                param_distributions=param_dist,
                n_iter=n_iter_search,
                cv=cv_strategy,
                scoring='accuracy',
                random_state=random_state,
                n_jobs=-1
            )
            search.fit(X_known_scaled, y_encoded)
            rf = search.best_estimator_
            print(f"\nBest parameters for {target_name}: {search.best_params_}")
            print(f"Best CV accuracy: {search.best_score_:.3f}")
        else:
            cv_scores = cross_val_score(rf, X_known_scaled, y_encoded, cv=cv_strategy, scoring='accuracy')
            print(f"CV accuracy for {target_name}: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            rf.fit(X_known_scaled, y_encoded)

        # === 5. Train-test split for evaluation ===
        X_train, X_test, y_train, y_test = train_test_split(
            X_known_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=random_state
        )

        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"\n✅ Accuracy for {target_name}: {acc:.3f}")

        # --- Confusion matrix ---
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.inverse_transform(np.unique(y_test)))
        disp.plot(cmap='Blues', xticks_rotation=45)
        plt.title(f"Confusion matrix for {target_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"ConfusionMatrix_{target_name}.png"), dpi=300)
        plt.close()

        # Normalized version
        cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
        disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=le.inverse_transform(np.unique(y_test)))
        disp_norm.plot(cmap='Blues', xticks_rotation=45)
        plt.title(f"Normalized confusion matrix for {target_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"ConfusionMatrix_{target_name}_normalized.png"), dpi=300)
        plt.close()

        # --- SHAP analysis ---
        print("Computing SHAP values (may take some time)...")
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test)

        shap.summary_plot(shap_values, X_test, feature_names=common_columns, show=False)
        plt.title(f"SHAP summary for {target_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"SHAP_summary_{target_name}.png"), dpi=300)
        plt.close()

        # === 6. Final training on all data ===
        rf.fit(X_known_scaled, y_encoded)
        models[target_name] = rf

        # === 7. Predict unknowns ===
        proba = rf.predict_proba(X_unknown_scaled)
        pred_idx = np.argmax(proba, axis=1)
        pred_labels = le.inverse_transform(pred_idx)
        confidence = np.max(proba, axis=1)

        predictions[target_name] = pred_labels
        confidences[f"{target_name}_confidence"] = confidence.round(2)

        # === 8. Save model ===
        joblib.dump(rf, f"{target_name}_balanced_model.joblib")

    # === 9. Export results ===
    results = pd.DataFrame(predictions)
    for conf_col, values in confidences.items():
        results[conf_col] = values
    if 'Samples' in data_unknown.columns:
        results['Samples'] = data_unknown['Samples']

    results.to_excel(output_path, index=False)
    print(f"\n✅ Predictions saved to: {output_path}")
    print(f"📊 All plots saved to: {plots_dir}")

    return results


# Example usage:
predictions = GLORIA_v5(
    database_path='GLORIA/Database.xlsx',
    data_unknown_path='Cluny/2024-09-12/Cluny_propre/Cluny_total_calibrated.xlsx',
    target_columns=['Century', 'Global region'],
    optimize_hyperparams=True,
    resample_method="smote",  # Options: None, "smote", "undersample"
    n_iter_search=30
)


#%%%% GLORIA V6



def GLORIA_v6(
    database_path: str,
    data_unknown_path: str,
    target_columns: list[str],
    optimize_hyperparams: bool = True,
    resample_method: str | None = None,  # None, "smote", "undersample"
    n_iter_search: int = 25,
    cv_folds: int = 5,
    random_state: int = 42,
    plots_dir: str = "Results/Plots",
    models_dir: str = "Results/Models",
    profiles_dir: str = "Results/Profiles"
):
    """
    GLORIA_v6 — Analysis & interpretability extension

    New features:
    - Confusion matrices (raw + normalized) saved per target
    - SHAP summary plots (if shap installed) saved per target
    - Feature importance barplot (rf.feature_importances_) saved
    - Permutation importance fallback saved
    - Export representative tree (text + plotted image)
    - Save mean/std profiles per class (CSV)
    - Save trained models with joblib

    Returns:
        pd.DataFrame: predictions for unknowns (with confidence columns)
    """

    # === Prepare folders ===
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(profiles_dir, exist_ok=True)

    # === 1. Load data ===
    data_known = pd.read_excel(database_path)
    data_unknown = pd.read_excel(data_unknown_path)
    output_path = data_unknown_path.split('.xlsx')[0] + '_v6.xlsx'

    exclude_cols = target_columns + [
        'ID', 'Reference', 'ID(Ref)', 'Site', 'City', 'Country', 'Region',
        'Global region', 'Form', 'Data Method?', 'Colour', 'Mg/Ca', 'Mg',
        'Date - Early', 'Date - Mean', 'Date - Late', 'Date',
        '(Na2O + MgO)/Sommes des autres'
    ]

    feature_columns = [c for c in data_known.columns if c not in exclude_cols]
    common_columns = [c for c in feature_columns if c in data_unknown.columns]

    if len(common_columns) == 0:
        raise ValueError("Aucune colonne commune entre database et unknown. Vérifie exclude_cols et fichiers.")

    X_known = data_known[common_columns].copy()
    X_unknown = data_unknown[common_columns].copy()

    # Clean common non-numeric tokens
    replace_map = {'': 0, 'REF': 0, '#VALUE!': 0, '-': 0, '< LOD': 0, '<LOD': 0, np.nan: 0}
    X_known.replace(replace_map, inplace=True)
    X_unknown.replace(replace_map, inplace=True)

    X_known = X_known.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_unknown = X_unknown.apply(pd.to_numeric, errors='coerce').fillna(0)

    models = {}
    label_encoders = {}
    predictions = {}
    confidences = {}

    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # === Loop per target ===
    for target_name in target_columns:
        print(f"\n{'='*60}\nTraining analysis for target: {target_name}\n{'='*60}")

        # scale anew per target
        scaler = StandardScaler()
        X_known_scaled = scaler.fit_transform(X_known)
        X_unknown_scaled = scaler.transform(X_unknown)

        # encode target labels
        if target_name not in data_known.columns:
            raise ValueError(f"Target column '{target_name}' not present in database.")
        y = data_known[target_name].copy()
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        label_encoders[target_name] = le

        # sanity check
        assert X_known_scaled.shape[0] == len(y_encoded), (
            f"Mismatch X/y sizes for {target_name}: X={X_known_scaled.shape[0]} vs y={len(y_encoded)}"
        )

        # print class distribution
        unique, counts = np.unique(y_encoded, return_counts=True)
        print("Original class distribution:")
        for cls_idx, cnt in zip(unique, counts):
            print(f"  {le.inverse_transform([cls_idx])[0]}: {cnt}")

        # optional resample
        if resample_method == "smote":
            if len(np.unique(y_encoded)) < 2:
                print(f"Skipping SMOTE for {target_name} (one class).")
            else:
                smote = SMOTE(random_state=random_state)
                X_known_scaled_res, y_encoded_res = smote.fit_resample(X_known_scaled, y_encoded)
                X_known_scaled, y_encoded = X_known_scaled_res, y_encoded_res
        elif resample_method == "undersample":
            rus = RandomUnderSampler(random_state=random_state)
            X_known_scaled, y_encoded = rus.fit_resample(X_known_scaled, y_encoded)

        # print post-resample distribution
        if resample_method:
            unique, counts = np.unique(y_encoded, return_counts=True)
            print("Post-resample distribution:")
            for cls_idx, cnt in zip(unique, counts):
                print(f"  {le.inverse_transform([cls_idx])[0]}: {cnt}")

        # === Model definition & hyperparam search ===
        rf = RandomForestClassifier(random_state=random_state, class_weight='balanced')

        if optimize_hyperparams:
            param_dist = {
                'n_estimators': np.arange(300, 601, 50),
                'max_depth': [30, 40, 50],
                'min_samples_split': np.arange(5, 11),
                'min_samples_leaf': [1],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
            search = RandomizedSearchCV(
                rf, param_distributions=param_dist,
                n_iter=n_iter_search, cv=cv_strategy, scoring='accuracy',
                random_state=random_state, n_jobs=-1
            )
            search.fit(X_known_scaled, y_encoded)
            rf = search.best_estimator_
            print(f"Best params for {target_name}: {search.best_params_}")
            print(f"Best CV acc: {search.best_score_:.3f}")
        else:
            cv_scores = cross_val_score(rf, X_known_scaled, y_encoded, cv=cv_strategy, scoring='accuracy')
            print(f"CV accuracy (no opt) for {target_name}: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            rf.fit(X_known_scaled, y_encoded)

        # === Train/test split for evaluation and interpretability ===
        X_train, X_test, y_train, y_test = train_test_split(
            X_known_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=random_state
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Evaluation accuracy on hold-out for {target_name}: {acc:.3f}")

        # --- Confusion matrices ---
        labels_unique = np.unique(y_test)
        display_labels = le.inverse_transform(labels_unique)

        cm = confusion_matrix(y_test, y_pred, labels=labels_unique)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(cmap='Blues', xticks_rotation=45)
        plt.title(f"Confusion matrix - {target_name}")
        plt.tight_layout()
        fn_cm = os.path.join(plots_dir, f"Confusion_{target_name}.png")
        plt.savefig(fn_cm, dpi=300)
        plt.close()

        # --- Feature importances (RF built-in) ---
        feat_imp = rf.feature_importances_
        fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(common_columns))))
        order = np.argsort(feat_imp)[::-1]
        ax.barh(np.array(common_columns)[order], feat_imp[order])
        ax.invert_yaxis()
        ax.set_xlabel("Feature importance (mean decrease impurity)")
        ax.set_title(f"RF feature importances - {target_name}")
        plt.tight_layout()
        fn_imp = os.path.join(plots_dir, f"FeatureImportances_{target_name}.png")
        plt.savefig(fn_imp, dpi=300)
        plt.close()

        # --- Export representative tree (text + plot) ---
        try:
            tree0 = rf.estimators_[0]
            thresholds_orig = scaler.inverse_transform(np.array([tree0.tree_.threshold]).T).flatten()
            tree0.tree_.threshold[:] = thresholds_orig
            classes_orig = le.classes_.astype(str)

            tree_text = export_text(tree0, feature_names=list(common_columns))
            with open(os.path.join(plots_dir, f"Tree0_rules_{target_name}.txt"), "w", encoding='utf-8') as f:
                f.write(tree_text)

            # plot tree (may be large)
            fig, ax = plt.subplots(figsize=(20, 12))
            plot_tree(tree0, feature_names=common_columns, class_names=classes_orig, filled=True, max_depth=4, ax=ax)
            plt.title(f"Representative tree (depth<=4) - {target_name}")
            plt.tight_layout()
            fn_tree = os.path.join(plots_dir, f"Tree0_plot_{target_name}.png")
            plt.savefig(fn_tree, dpi=300)
            plt.close()
        except Exception as e:
            print("Could not export/plot tree:", e)

        # --- SHAP analysis ---

            print("Computing SHAP values (this can be slow for large trees)...")
            # TreeExplainer is efficient for tree models
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_test)

            # summary plot (beeswarm)
            shap.summary_plot(shap_values, X_test, feature_names=common_columns, show=False)
            plt.title(f"SHAP summary (beeswarm) - {target_name}")
            plt.tight_layout()
            fn_shap = os.path.join(plots_dir, f"SHAP_summary_{target_name}.png")
            plt.savefig(fn_shap, dpi=300)
            plt.close()

            # bar plot (mean |shap|)
            try:
                shap.summary_plot(shap_values, X_test, feature_names=common_columns, plot_type="bar", show=False)
                plt.title(f"SHAP mean(|value|) - {target_name}")
                plt.tight_layout()
                fn_shap_bar = os.path.join(plots_dir, f"SHAP_bar_{target_name}.png")
                plt.savefig(fn_shap_bar, dpi=300)
                plt.close()
            except Exception:
                pass

        # --- Save class profiles (mean & std) ---
        try:
            # original classes labels and numeric dataframe
            df_known_feats = pd.DataFrame(X_known, columns=common_columns)
            df_known_feats[target_name] = data_known[target_name].values
            profiles_mean = df_known_feats.groupby(target_name)[common_columns].mean()
            profiles_std = df_known_feats.groupby(target_name)[common_columns].std()
            profiles_mean.to_csv(os.path.join(profiles_dir, f"Profiles_mean_{target_name}.csv"))
            profiles_std.to_csv(os.path.join(profiles_dir, f"Profiles_std_{target_name}.csv"))
        except Exception as e_prof:
            print("Could not compute/save profiles:", e_prof)


        # === Predictions on unknowns ===
        proba = rf.predict_proba(X_unknown_scaled)
        pred_idx = np.argmax(proba, axis=1)
        pred_labels = le.inverse_transform(pred_idx)
        confidence = np.max(proba, axis=1)

        predictions[target_name] = pred_labels
        confidences[f"{target_name}_confidence"] = np.round(confidence, 3)

        print(f"Saved plots for {target_name} in {plots_dir}")
        print(f"Saved model and encoders for {target_name} in {models_dir}")

    # === 9. Export combined results ===
    results = pd.DataFrame(predictions)
    for conf_col, values in confidences.items():
        results[conf_col] = values
    if 'Samples' in data_unknown.columns:
        results['Samples'] = data_unknown['Samples']

    results.to_excel(output_path, index=False)
    print(f"\nAll predictions saved to: {output_path}")
    print(f"All plots saved to: {plots_dir}")
    print(f"All models saved to: {models_dir}")
    print(f"All profiles saved to: {profiles_dir}")

    return results

# Example call (adapt paths as needed)
predictions = GLORIA_v6(
    database_path='Database.xlsx',
    data_unknown_path='Samples_to_train.xlsx',
    target_columns=['Century', 'Global region'],
    optimize_hyperparams=True,
    resample_method="smote",
    n_iter_search=30
)


#%%%


def summarize_forest_rules(rf, scaler, le, feature_names, target_name, plots_dir, min_support=0.05):
    """
    Résume les règles dominantes d'une RandomForest entraînée.
    Exporte un .txt et un .csv listant les règles les plus fréquentes.
    """
    from collections import Counter, defaultdict

    rules_counter = Counter()
    class_counter = defaultdict(Counter)
    n_trees = len(rf.estimators_)
    classes_orig = le.classes_.astype(str)

    # --- Parcourt chaque arbre ---
    for tree_idx, tree_model in enumerate(rf.estimators_):
        tree_ = tree_model.tree_
        features = [feature_names[i] if i != -2 else "Leaf" for i in tree_.feature]

        def recurse(node, current_rule):
            if tree_.feature[node] != -2:  # nœud interne
                feat_idx = tree_.feature[node]
                thr_std = tree_.threshold[node]
                mean_val = scaler.mean_[feat_idx]
                std_val = np.sqrt(scaler.var_[feat_idx])
                thr_real = thr_std * std_val + mean_val

                recurse(tree_.children_left[node], current_rule + [f"{features[node]} <= {thr_real:.3f}"])
                recurse(tree_.children_right[node], current_rule + [f"{features[node]} > {thr_real:.3f}"])
            else:
                class_idx = np.argmax(tree_.value[node][0])
                cls = classes_orig[class_idx]
                rule_text = " AND ".join(current_rule)
                rules_counter[rule_text] += 1
                class_counter[rule_text][cls] += 1

        recurse(0, [])

    # --- Synthèse ---
    total_trees = n_trees
    rules_summary = []
    for rule, count in rules_counter.items():
        freq = count / total_trees
        if freq >= min_support:
            cls_major = class_counter[rule].most_common(1)[0][0]
            cls_ratio = class_counter[rule][cls_major] / count
            rules_summary.append((freq, cls_major, cls_ratio, rule))

    # Tri par fréquence décroissante
    rules_summary.sort(reverse=True, key=lambda x: x[0])

    # --- Export TXT ---
    txt_path = os.path.join(plots_dir, f"Dominant_Rules_{target_name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"=== Dominant rules for target: {target_name} ===\n")
        f.write(f"Total trees analyzed: {n_trees}\n")
        f.write(f"Minimum support: {min_support*100:.1f}%\n\n")
        for freq, cls, cls_ratio, rule in rules_summary:
            f.write(f"[{freq*100:.1f}% of trees] → class {cls} ({cls_ratio*100:.1f}% agreement)\n")
            f.write(f"    IF {rule}\n\n")

    # --- Export CSV ---
    df_summary = pd.DataFrame(rules_summary, columns=["Frequency", "Class", "Agreement", "Rule"])
    csv_path = os.path.join(plots_dir, f"Dominant_Rules_{target_name}.csv")
    df_summary.to_csv(csv_path, index=False)

    print(f"✅ Dominant rules exported for {target_name}:")
    print(f"   - {txt_path}")
    print(f"   - {csv_path}")



#%%% GLORIA V7
def GLORIA_v7(
    database_path: str,
    data_unknown_path: str,
    target_columns: list[str],
    optimize_hyperparams: bool = True,
    resample_method: str | None = None,  # None, "smote", "undersample"
    n_iter_search: int = 25,
    cv_folds: int = 5,
    random_state: int = 42,
    plots_dir: str = "Results/Plots",
    models_dir: str = "Results/Models",
    profiles_dir: str = "Results/Profiles"
):
    """
    GLORIA_v7 — RandomForest multi-cible avec analyses, profils, règles et interprétabilité.
    """

    # === Préparation des dossiers ===
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(profiles_dir, exist_ok=True)

    # === 1. Chargement des données ===
    data_known = pd.read_excel(database_path)
    data_unknown = pd.read_excel(data_unknown_path)
    output_path = data_unknown_path.split('.xlsx')[0] + '_v7.xlsx'

    exclude_cols = target_columns + [
        'ID', 'Reference', 'ID(Ref)', 'Site', 'City', 'Country', 'Region',
        'Global region', 'Form', 'Data Method?', 'Colour', 'Mg/Ca', 'Mg',
        'Date - Early', 'Date - Mean', 'Date - Late', 'Date',
        '(Na2O + MgO)/Sommes des autres'
    ]

    feature_columns = [c for c in data_known.columns if c not in exclude_cols]
    common_columns = [c for c in feature_columns if c in data_unknown.columns]
    if len(common_columns) == 0:
        raise ValueError("Aucune colonne commune entre database et unknown.")

    X_known = data_known[common_columns].copy()
    X_unknown = data_unknown[common_columns].copy()

    # Nettoyage
    replace_map = {'': 0, 'REF': 0, '#VALUE!': 0, '-': 0, '< LOD': 0, '<LOD': 0, np.nan: 0}
    X_known.replace(replace_map, inplace=True)
    X_unknown.replace(replace_map, inplace=True)
    X_known = X_known.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_unknown = X_unknown.apply(pd.to_numeric, errors='coerce').fillna(0)

    models = {}
    label_encoders = {}
    predictions = {}
    confidences = {}

    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # === Boucle par target ===
    for target_name in target_columns:
        print(f"\n{'='*60}\nTraining analysis for target: {target_name}\n{'='*60}")

        scaler = StandardScaler()
        X_known_scaled = scaler.fit_transform(X_known)
        X_unknown_scaled = scaler.transform(X_unknown)

        if target_name not in data_known.columns:
            raise ValueError(f"Target column '{target_name}' not found.")
        y = data_known[target_name].copy()
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        label_encoders[target_name] = le

        # Option de rééchantillonnage
        if resample_method == "smote" and len(np.unique(y_encoded)) > 1:
            smote = SMOTE(random_state=random_state)
            X_known_scaled, y_encoded = smote.fit_resample(X_known_scaled, y_encoded)
        elif resample_method == "undersample":
            rus = RandomUnderSampler(random_state=random_state)
            X_known_scaled, y_encoded = rus.fit_resample(X_known_scaled, y_encoded)

        # === Modèle & optimisation ===
        rf = RandomForestClassifier(random_state=random_state, class_weight='balanced')
        if optimize_hyperparams:
            param_dist = {
                'n_estimators': np.arange(300, 601, 50),
                'max_depth': [30, 40, 50],
                'min_samples_split': np.arange(5, 11),
                'min_samples_leaf': [1],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
            search = RandomizedSearchCV(
                rf, param_distributions=param_dist,
                n_iter=n_iter_search, cv=cv_strategy, scoring='accuracy',
                random_state=random_state, n_jobs=-1
            )
            search.fit(X_known_scaled, y_encoded)
            rf = search.best_estimator_
            print(f"Best params for {target_name}: {search.best_params_}")
        else:
            rf.fit(X_known_scaled, y_encoded)

        # === Évaluation ===
        X_train, X_test, y_train, y_test = train_test_split(
            X_known_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=random_state
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Hold-out accuracy for {target_name}: {acc:.3f}")

        # === Matrice de confusion ===
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
        disp.plot(cmap='Blues', xticks_rotation=45)
        plt.title(f"Confusion matrix - {target_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"Confusion_{target_name}.png"), dpi=300)
        plt.close()

        # === Export des arbres ===
        try:
            tree0 = rf.estimators_[0]
            classes_orig = le.classes_.astype(str)
            thresholds_scaled = tree0.tree_.threshold.copy()
            thresholds_real = []
            for feat_idx, thr in zip(tree0.tree_.feature, thresholds_scaled):
                if feat_idx == -2:
                    thresholds_real.append(-2)
                else:
                    mean_val = scaler.mean_[feat_idx]
                    std_val = np.sqrt(scaler.var_[feat_idx])
                    thr_real = thr * std_val + mean_val
                    thresholds_real.append(thr_real)
            tree0.tree_.threshold[:] = thresholds_real

            tree_text = export_text(tree0, feature_names=list(common_columns), decimals=3, show_weights=True)
            for i, cls in enumerate(classes_orig):
                tree_text = tree_text.replace(f"class: {i}", f"class: {cls}")

            txt_path = os.path.join(plots_dir, f"Tree0_rules_real_{target_name}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(tree_text)

            fig, ax = plt.subplots(figsize=(20, 12))
            plot_tree(tree0, feature_names=common_columns, class_names=classes_orig, filled=True, max_depth=4, ax=ax)
            plt.title(f"Representative tree (depth<=4) - {target_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"Tree0_plot_real_{target_name}.png"), dpi=300)
            plt.close()
        except Exception as e:
            print("Could not export tree:", e)

        # === Export de tous les arbres ===
        try:
            all_rules_path = os.path.join(plots_dir, f"All_Trees_Rules_real_{target_name}.txt")
            with open(all_rules_path, "w", encoding="utf-8") as f_out:
                f_out.write(f"=== RANDOM FOREST RULES for target: {target_name} ===\n\n")
                for idx, tree_model in enumerate(rf.estimators_):
                    f_out.write(f"\n===== TREE {idx+1}/{len(rf.estimators_)} =====\n")
                    thresholds_scaled = tree_model.tree_.threshold.copy()
                    thresholds_real = []
                    for feat_idx, thr in zip(tree_model.tree_.feature, thresholds_scaled):
                        if feat_idx == -2:
                            thresholds_real.append(-2)
                        else:
                            mean_val = scaler.mean_[feat_idx]
                            std_val = np.sqrt(scaler.var_[feat_idx])
                            thr_real = thr * std_val + mean_val
                            thresholds_real.append(thr_real)
                    tree_model.tree_.threshold[:] = thresholds_real
                    tree_text = export_text(tree_model, feature_names=list(common_columns), decimals=3, show_weights=True)
                    for i, cls in enumerate(classes_orig):
                        tree_text = tree_text.replace(f"class: {i}", f"class: {cls}")
                    f_out.write(tree_text)
            print(f"✅ Exported all tree rules to {all_rules_path}")
        except Exception as e:
            print("Could not export all trees:", e)

        # === Règles dominantes ===
        summarize_forest_rules(rf, scaler, le, common_columns, target_name, plots_dir, min_support=0.05)

        # === Prédictions sur échantillons inconnus ===
        proba = rf.predict_proba(X_unknown_scaled)
        pred_idx = np.argmax(proba, axis=1)
        pred_labels = le.inverse_transform(pred_idx)
        confidence = np.max(proba, axis=1)

        predictions[target_name] = pred_labels
        confidences[f"{target_name}_confidence"] = np.round(confidence, 3)

    # === Export final ===
    results = pd.DataFrame(predictions)
    for conf_col, values in confidences.items():
        results[conf_col] = values
    if 'Samples' in data_unknown.columns:
        results['Samples'] = data_unknown['Samples']

    results.to_excel(output_path, index=False)
    print(f"\nAll predictions saved to: {output_path}")
    print(f"All plots & rule exports saved to: {plots_dir}")

    return results


predictions = GLORIA_v7(
    database_path='Database.xlsx',
    data_unknown_path='Samples_to_train.xlsx',
    target_columns=['Century', 'Global region'],
    optimize_hyperparams=True,
    resample_method="smote",
    n_iter_search=30
)

#%%% GLORIA V5_1

def GLORIA_v5_1(
    database_path: str,
    data_unknown_path: str,
    target_columns: list[str],   # user-defined target columns to combine
    optimize_hyperparams: bool = True,
    resample_method: str | None = None,  # Options: None, "smote", "undersample"
    n_iter_search: int = 25,
    cv_folds: int = 5,
    random_state: int = 42
):
    """
    GLORIA_v5_combined — Joint prediction on multiple target columns (e.g. Century + Global region)
    Version: October 2025

    Features:
    - Combines user-specified target columns into one composite target
    - Includes confusion matrices and SHAP feature importance plots
    - Automatically saves plots and predictions
    """

    # === 1. Load data ===
    data_known = pd.read_excel(database_path)
    data_unknown = pd.read_excel(data_unknown_path)
    output_path = data_unknown_path.split('.xlsx')[0] + '_v5_1.xlsx'

    # === 2. Validate target columns ===
    for col in target_columns:
        if col not in data_known.columns:
            raise ValueError(f"Target column '{col}' not found in database.")
    print(f"🔗 Combining target columns: {', '.join(target_columns)}")

    # Create composite target (e.g., "18th_Europe")
    composite_target_name = "_".join(target_columns)
    data_known[composite_target_name] = data_known[target_columns].astype(str).apply(
        lambda row: "_".join(row.values.astype(str)).strip(), axis=1
    )

    # === 3. Feature selection ===
    exclude_cols = target_columns + [
        composite_target_name, 'ID', 'Reference', 'ID(Ref)', 'Site', 'City', 'Country',
        'Region', 'Global region', 'Form', 'Data Method?', 'Colour', 'Mg/Ca', 'Mg',
        'Date - Early', 'Date - Mean', 'Date - Late', 'Date',
        '(Na2O + MgO)/Sommes des autres'
    ]

    feature_columns = [c for c in data_known.columns if c not in exclude_cols]
    common_columns = [c for c in feature_columns if c in data_unknown.columns]

    X_known = data_known[common_columns].copy()
    X_unknown = data_unknown[common_columns].copy()

    # Replace non-numeric or invalid values
    replace_map = {'': 0, 'REF': 0, '#VALUE!': 0, '-': 0, '< LOD': 0, '<LOD': 0, np.nan: 0}
    X_known.replace(replace_map, inplace=True)
    X_unknown.replace(replace_map, inplace=True)

    X_known = X_known.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_unknown = X_unknown.apply(pd.to_numeric, errors='coerce').fillna(0)

    y = data_known[composite_target_name].copy()

    # === 4. Create output folders ===
    plots_dir = os.path.join("Results", "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    # === 5. Encode and scale ===
    scaler = StandardScaler()
    X_known_scaled = scaler.fit_transform(X_known)
    X_unknown_scaled = scaler.transform(X_unknown)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"\nNumber of combined classes: {len(le.classes_)}")
    unique, counts = np.unique(y_encoded, return_counts=True)
    print("Initial class distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"  {le.inverse_transform([cls])[0]}: {cnt} samples")

    # === 6. Optional resampling ===
    # Remove classes that are too small for SMOTE
    min_class_size = 6  # SMOTE requires at least 6 samples (k_neighbors=5 + 1)
    class_counts = pd.Series(y_encoded).value_counts()

    small_classes = class_counts[class_counts < min_class_size].index.tolist()
    if small_classes:
        removed_labels = le.inverse_transform(small_classes)
        print(f"\n⚠️  Removing {len(small_classes)} rare classes (less than {min_class_size} samples):")
        for lbl in removed_labels:
            print(f"   - {lbl}")
        mask = ~pd.Series(y_encoded).isin(small_classes)
        X_known_scaled = X_known_scaled[mask]
        y_encoded = y_encoded[mask]
        
    if resample_method == "smote":
        print("\n🧬 Applying SMOTE oversampling...")
        smote = SMOTE(random_state=random_state)
        X_known_scaled, y_encoded = smote.fit_resample(X_known_scaled, y_encoded)
    elif resample_method == "undersample":
        print("\n⚖️  Applying random undersampling...")
        rus = RandomUnderSampler(random_state=random_state)
        X_known_scaled, y_encoded = rus.fit_resample(X_known_scaled, y_encoded)

    if resample_method:
        unique, counts = np.unique(y_encoded, return_counts=True)
        print("Balanced class distribution:")
        for cls, cnt in zip(unique, counts):
            print(f"  {le.inverse_transform([cls])[0]}: {cnt} samples")

    # === 7. Model definition and optimization ===
    rf = RandomForestClassifier(random_state=random_state, class_weight='balanced')
    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    if optimize_hyperparams:
        print("\n🔎 Optimizing hyperparameters (RandomizedSearchCV)...")
        param_dist = {
            'n_estimators': np.arange(300, 501, 50),
            'max_depth': [30, 40, 50],
            'min_samples_split': np.arange(2, 11),
            'min_samples_leaf': [1],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        search = RandomizedSearchCV(
            rf, param_distributions=param_dist, n_iter=n_iter_search,
            cv=cv_strategy, scoring='accuracy', random_state=random_state, n_jobs=-1
        )
        search.fit(X_known_scaled, y_encoded)
        rf = search.best_estimator_
        print(f"\n✅ Best parameters: {search.best_params_}")
        print(f"Best CV accuracy: {search.best_score_:.3f}")
    else:
        cv_scores = cross_val_score(rf, X_known_scaled, y_encoded, cv=cv_strategy, scoring='accuracy')
        print(f"CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        rf.fit(X_known_scaled, y_encoded)

    # === 8. Evaluation with train/test split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X_known_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=random_state
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n📈 Combined model accuracy: {acc:.3f}")

    # --- Confusion matrices ---
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.inverse_transform(np.unique(y_test)))
    disp.plot(cmap='Blues', xticks_rotation=90)
    plt.title(f"Confusion Matrix ({composite_target_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"ConfusionMatrix_{composite_target_name}.png"), dpi=300)
    plt.close()

    cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=le.inverse_transform(np.unique(y_test)))
    disp_norm.plot(cmap='Blues', xticks_rotation=90)
    plt.title(f"Normalized Confusion Matrix ({composite_target_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"ConfusionMatrix_{composite_target_name}_normalized.png"), dpi=300)
    plt.close()

    # --- SHAP feature importance ---
    # print("\n💡 Computing SHAP values...")
    # explainer = shap.TreeExplainer(rf)
    # shap_values = explainer.shap_values(X_test)
    # shap.summary_plot(shap_values, X_test, feature_names=common_columns, show=False)
    # plt.title(f"SHAP Summary ({composite_target_name})")
    # plt.tight_layout()
    # plt.savefig(os.path.join(plots_dir, f"SHAP_summary_{composite_target_name}.png"), dpi=300)
    # plt.close()

    # === 9. Final training on all known data ===
    rf.fit(X_known_scaled, y_encoded)
    joblib.dump(rf, f"{composite_target_name}_model.joblib")

    # === 10. Predict on unknown data ===
    proba = rf.predict_proba(X_unknown_scaled)
    pred_idx = np.argmax(proba, axis=1)
    pred_labels = le.inverse_transform(pred_idx)
    confidence = np.max(proba, axis=1)

    results = pd.DataFrame({
        composite_target_name: pred_labels,
        f"{composite_target_name}_confidence": confidence.round(2)
    })

    # Split combined prediction into separate columns
    results[[f"Pred_{col}" for col in target_columns]] = results[composite_target_name].str.split('_', n=len(target_columns)-1, expand=True)

    if 'Samples' in data_unknown.columns:
        results['Samples'] = data_unknown['Samples']

    # === 11. Save results ===
    results.to_excel(output_path, index=False)
    print(f"\n✅ Predictions saved to: {output_path}")
    print(f"📊 All plots saved to: {plots_dir}")

    return results

predictions = GLORIA_v5_1(
    database_path='GLORIA/Database.xlsx',
    data_unknown_path='GLORIA/Samples_to_train.xlsx',
    target_columns=['Century', 'Global region'],  # user chooses which targets to combine
    optimize_hyperparams=True,
    resample_method="smote",
    n_iter_search=30
)



#%%% GLORIA V5_2


def GLORIA_v5_2(
    database_path: str,
    data_unknown_path: str,
    target_columns: list[str],
    optimize_hyperparams: bool = True,
    resample_method: str | None = None,  # Options: None, "smote", "undersample"
    n_iter_search: int = 25,
    cv_folds: int = 5,
    random_state: int = 42
):
    """
    GLORIA_v5 — Enhanced version (Oct. 2025)
    Focuses only on the window glass for the database
    """

    # === 1. Load and clean data ===
    data_known = pd.read_excel(database_path, sheet_name="Windows")
    data_unknown = pd.read_excel(data_unknown_path)
    output_path = data_unknown_path.split('.xlsx')[0] + '_v5_2.xlsx'

    exclude_cols = target_columns + [
        'ID', 'Reference', 'ID(Ref)', 'Site', 'City', 'Country', 'Region',
        'Global region', 'Form', 'Data Method?', 'Colour', 'Mg/Ca', 'Mg',
        'Date - Early', 'Date - Mean', 'Date - Late', 'Date',
        '(Na2O + MgO)/Sommes des autres'
    ]

    feature_columns = [c for c in data_known.columns if c not in exclude_cols]
    common_columns = [c for c in feature_columns if c in data_unknown.columns]

    X_known = data_known[common_columns].copy()
    X_unknown = data_unknown[common_columns].copy()

    replace_map = {'': 0, 'REF': 0, '#VALUE!': 0, '-': 0, '< LOD': 0, '<LOD': 0, np.nan: 0}
    X_known.replace(replace_map, inplace=True)
    X_unknown.replace(replace_map, inplace=True)

    X_known = X_known.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_unknown = X_unknown.apply(pd.to_numeric, errors='coerce').fillna(0)

    models, label_encoders, predictions, confidences = {}, {}, {}, {}

    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # === Create output folders ===
    plots_dir = os.path.join("Results", "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    # === 2. Training loop per target ===
    for target_name in target_columns:
        print(f"\n{'='*60}")
        print(f"--- Training model for target: {target_name} ---")

        scaler = StandardScaler()
        X_known_scaled = scaler.fit_transform(X_known)
        X_unknown_scaled = scaler.transform(X_unknown)

        # Encode target
        y = data_known[target_name].copy()
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        label_encoders[target_name] = le

        # Vérification cohérence
        assert X_known_scaled.shape[0] == len(y_encoded), (
            f"Shape mismatch for {target_name}: X_known_scaled={X_known_scaled.shape[0]}, y_encoded={len(y_encoded)}"
        )

        # Distribution initiale
        unique, counts = np.unique(y_encoded, return_counts=True)
        print("Original class distribution:")
        for cls, cnt in zip(unique, counts):
            print(f"  {le.inverse_transform([cls])[0]}: {cnt} samples")

        # === 3. Optional resampling ===
        if resample_method == "smote":
            if len(np.unique(y_encoded)) < 2:
                print(f"⚠️  Skipping SMOTE for {target_name} (only one class present).")
            else:
                print("\nApplying SMOTE oversampling...")
                smote = SMOTE(random_state=random_state)
                X_known_scaled, y_encoded = smote.fit_resample(X_known_scaled, y_encoded)
        elif resample_method == "undersample":
            print("\nApplying random undersampling...")
            rus = RandomUnderSampler(random_state=random_state)
            X_known_scaled, y_encoded = rus.fit_resample(X_known_scaled, y_encoded)

        # Distribution après rééchantillonnage
        if resample_method:
            unique, counts = np.unique(y_encoded, return_counts=True)
            print("Balanced class distribution:")
            for cls, cnt in zip(unique, counts):
                print(f"  {le.inverse_transform([cls])[0]}: {cnt} samples")

        # === 4. Define and optimize model ===
        rf = RandomForestClassifier(random_state=random_state, class_weight='balanced')

        if optimize_hyperparams:
            param_dist = {
                'n_estimators': np.arange(300, 501, 50),
                'max_depth': [30, 40, 50],
                'min_samples_split': np.arange(2, 11),
                'min_samples_leaf': [1],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }

            search = RandomizedSearchCV(
                rf,
                param_distributions=param_dist,
                n_iter=n_iter_search,
                cv=cv_strategy,
                scoring='accuracy',
                random_state=random_state,
                n_jobs=-1
            )
            search.fit(X_known_scaled, y_encoded)
            rf = search.best_estimator_
            print(f"\nBest parameters for {target_name}: {search.best_params_}")
            print(f"Best CV accuracy: {search.best_score_:.3f}")
        else:
            cv_scores = cross_val_score(rf, X_known_scaled, y_encoded, cv=cv_strategy, scoring='accuracy')
            print(f"CV accuracy for {target_name}: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            rf.fit(X_known_scaled, y_encoded)

        # === 5. Train-test split for evaluation ===
        X_train, X_test, y_train, y_test = train_test_split(
            X_known_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=random_state
        )

        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"\n✅ Accuracy for {target_name}: {acc:.3f}")

        # --- Confusion matrix ---
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.inverse_transform(np.unique(y_test)))
        disp.plot(cmap='Blues', xticks_rotation=45)
        plt.title(f"Confusion matrix for {target_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"ConfusionMatrix_{target_name}.png"), dpi=300)
        plt.close()

        # Normalized version
        cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
        disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=le.inverse_transform(np.unique(y_test)))
        disp_norm.plot(cmap='Blues', xticks_rotation=45)
        plt.title(f"Normalized confusion matrix for {target_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"ConfusionMatrix_{target_name}_normalized.png"), dpi=300)
        plt.close()

        # --- SHAP analysis ---
        # print("Computing SHAP values (may take some time)...")
        # explainer = shap.TreeExplainer(rf)
        # shap_values = explainer.shap_values(X_test)

        # shap.summary_plot(shap_values, X_test, feature_names=common_columns, show=False)
        # plt.title(f"SHAP summary for {target_name}")
        # plt.tight_layout()
        # plt.savefig(os.path.join(plots_dir, f"SHAP_summary_{target_name}.png"), dpi=300)
        # plt.close()

        # === 6. Final training on all data ===
        rf.fit(X_known_scaled, y_encoded)
        models[target_name] = rf

        # === 7. Predict unknowns ===
        proba = rf.predict_proba(X_unknown_scaled)
        pred_idx = np.argmax(proba, axis=1)
        pred_labels = le.inverse_transform(pred_idx)
        confidence = np.max(proba, axis=1)

        predictions[target_name] = pred_labels
        confidences[f"{target_name}_confidence"] = confidence.round(2)

        # === 8. Save model ===
        joblib.dump(rf, f"{target_name}_balanced_model.joblib")

    # === 9. Export results ===
    results = pd.DataFrame(predictions)
    for conf_col, values in confidences.items():
        results[conf_col] = values
    if 'Samples' in data_unknown.columns:
        results['Samples'] = data_unknown['Samples']

    results.to_excel(output_path, index=False)
    print(f"\n✅ Predictions saved to: {output_path}")
    print(f"📊 All plots saved to: {plots_dir}")

    return results


# Example usage:
predictions = GLORIA_v5_2(
    database_path='GLORIA/Database.xlsx',
    data_unknown_path='GLORIA/Samples_to_train.xlsx',
    target_columns=['Century', 'Global region'],
    optimize_hyperparams=True,
    resample_method="smote",  # Options: None, "smote", "undersample"
    n_iter_search=30
)


#%%% GLORIA V8

def file_hash(path, chunk_size=8192):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def GLORIA_v8(
    database_path: str,
    data_unknown_path: str,
    target_columns: list[str],
    optimize_hyperparams: bool = True,
    resample_method: str | None = None,
    n_iter_search: int = 25,
    cv_folds: int = 5,
    random_state: int = 42
):
    """
    GLORIA_v8
    - Persistent models
    - Database hash checking
    - Optional retraining
    """

    print("\n================ GLORIA v8 ================\n")

    # === Paths ===
    models_dir = "GLORIA_models"
    plots_dir = os.path.join("Results", "Plots")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    metadata_path = os.path.join(models_dir, "metadata.json")
    output_path = data_unknown_path.replace(".xlsx", "_v8.xlsx")

    # === Load data ===
    data_known = pd.read_excel(database_path, sheet_name="Windows")
    data_unknown = pd.read_excel(data_unknown_path)

    # === Hash database ===
    current_hash = file_hash(database_path)
    reuse_models = False

    if os.path.exists(metadata_path):
        metadata = load_json(metadata_path)
        if metadata.get("database_hash") == current_hash:
            answer = input(
                "⚠️ Database unchanged.\n"
                "Reuse existing trained models? [y/n]: "
            ).strip().lower()
            reuse_models = (answer == "y")

    # === Feature selection ===
    exclude_cols = target_columns + [
        'ID', 'Reference', 'ID(Ref)', 'Site', 'City', 'Country',
        'Region', 'Global region', 'Form', 'Data Method?', 'Colour',
        'Mg/Ca', 'Mg', 'Date - Early', 'Date - Mean',
        'Date - Late', 'Date',
        '(Na2O + MgO)/Sommes des autres'
    ]

    feature_columns = [c for c in data_known.columns if c not in exclude_cols]
    common_columns = [c for c in feature_columns if c in data_unknown.columns]

    # === Cleaning ===
    replace_map = {'': 0, 'REF': 0, '#VALUE!': 0, '-':0, '<LOD': 0, '< LOD': 0, np.nan: 0}

    X_known = (
        data_known[common_columns]
        .replace(replace_map)
        .infer_objects(copy=False)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    X_unknown = (
        data_unknown[common_columns]
        .replace(replace_map)
        .infer_objects(copy=False)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    results, confidences = {}, {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # === Loop per target ===
    for target in target_columns:

        print(f"\n{'='*60}")
        print(f"Target: {target}")

        model_path = os.path.join(models_dir, f"{target}_model.joblib")

        # =====================================================
        # LOAD EXISTING MODEL
        # =====================================================
        if reuse_models and os.path.exists(model_path):

            print(f" Loading model for {target}")

            bundle = joblib.load(model_path)
            rf = bundle["model"]
            scaler = bundle["scaler"]
            le = bundle["label_encoder"]
            features = bundle["features"]

            X_unknown_scaled = scaler.transform(X_unknown[features])

        # =====================================================
        #  TRAIN NEW MODEL
        # =====================================================
        else:
            print("Training new model")

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_known)

            y = data_known[target]
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            # Optional resampling
            X_train = X_scaled.copy()
            y_train = y_encoded.copy()

            if resample_method == "smote":
                smote = SMOTE(random_state=random_state)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            elif resample_method == "undersample":
                rus = RandomUnderSampler(random_state=random_state)
                X_train, y_train = rus.fit_resample(X_train, y_train)

            rf = RandomForestClassifier(
                random_state=random_state,
                class_weight="balanced"
            )

            if optimize_hyperparams:
                param_dist = {
                    "n_estimators": np.arange(300, 501, 50),
                    "max_depth": [30, 40, 50],
                    "min_samples_split": np.arange(2, 11),
                    "min_samples_leaf": [1],
                    "max_features": ["sqrt", "log2", None],
                    "bootstrap": [True, False]
                }

                search = RandomizedSearchCV(
                    rf,
                    param_distributions=param_dist,
                    n_iter=n_iter_search,
                    cv=cv,
                    scoring="accuracy",
                    n_jobs=-1,
                    random_state=random_state
                )

                search.fit(X_train, y_train)
                rf = search.best_estimator_

                print(f"Best CV accuracy: {search.best_score_:.3f}")

            rf.fit(X_train, y_train)

            # Save model bundle
            joblib.dump(
                {
                    "model": rf,
                    "scaler": scaler,
                    "label_encoder": le,
                    "features": common_columns
                },
                model_path
            )

            X_unknown_scaled = scaler.transform(X_unknown)

        # =====================================================
        #  Prediction
        # =====================================================
        proba = rf.predict_proba(X_unknown_scaled)
        pred_idx = np.argmax(proba, axis=1)

        results[target] = le.inverse_transform(pred_idx)
        confidences[f"{target}_confidence"] = np.max(proba, axis=1).round(2)

    # === Save metadata ===
    if not reuse_models:
        save_json(
            metadata_path,
            {
                "database_hash": current_hash,
                "features": common_columns,
                "targets": target_columns,
                "created_on": datetime.now().isoformat()
            }
        )

    # === Export ===
    df_out = pd.DataFrame(results)
    for k, v in confidences.items():
        df_out[k] = v

    if "Samples" in data_unknown.columns:
        df_out["Samples"] = data_unknown["Samples"]

    df_out.to_excel(output_path, index=False)

    print(f"\nPredictions saved to: {output_path}")
    print("Models stored in:", models_dir)

    return df_out

predictions = GLORIA_v8(
    database_path='GLORIA/Database.xlsx',
    data_unknown_path='GLORIA/Samples_to_train.xlsx',
    target_columns=['Century', 'Global region'],
    optimize_hyperparams=True,
    resample_method="smote",  # Options: None, "smote", "undersample"
    n_iter_search=30
)


#%%% GLORIA_V9

def dataframe_hash(df: pd.DataFrame) -> str:
    """Hash only the meaningful content of a DataFrame."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def train_model_for_target(
    X: pd.DataFrame,
    y: pd.Series,
    target_name: str,
    common_columns: list[str],
    optimize_hyperparams: bool,
    resample_method: str | None,
    cv,
    n_iter_search: int,
    random_state: int
):
    """Train a model for a single target and return a reusable bundle."""

    print(f" Training model for {target_name}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # --- Class distribution check ---
    class_counts = np.bincount(y_encoded)
    min_samples = class_counts.min()

    # --- Optional resampling with guard ---
    X_train, y_train = X_scaled.copy(), y_encoded.copy()

    if resample_method == "smote":
        if min_samples < 10:
            print(f" SMOTE skipped for {target_name} (min class size = {min_samples})")
        else:
            print("Applying SMOTE...")
            smote = SMOTE(random_state=random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)

    elif resample_method == "undersample":
        rus = RandomUnderSampler(random_state=random_state)
        X_train, y_train = rus.fit_resample(X_train, y_train)

    rf = RandomForestClassifier(
        random_state=random_state,
        class_weight="balanced"
    )

    if optimize_hyperparams:
        param_dist = {
            "n_estimators": np.arange(300, 501, 50),
            "max_depth": [30, 40, 50],
            "min_samples_split": np.arange(2, 11),
            "min_samples_leaf": [1],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False]
        }

        search = RandomizedSearchCV(
            rf,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            cv=cv,
            scoring="f1_weighted",
            n_jobs=-1,
            random_state=random_state
        )

        search.fit(X_train, y_train)
        rf = search.best_estimator_

        print(f"Best CV f1_weighted: {search.best_score_:.3f}")

    rf.fit(X_train, y_train)

    return {
        "model": rf,
        "scaler": scaler,
        "label_encoder": le,
        "features": common_columns
    }

def predict_with_bundle(bundle, X_unknown: pd.DataFrame):
    """Apply a trained bundle to unknown data."""
    rf = bundle["model"]
    scaler = bundle["scaler"]
    le = bundle["label_encoder"]
    features = bundle["features"]

    X_scaled = scaler.transform(X_unknown[features])

    proba = rf.predict_proba(X_scaled)
    pred_idx = np.argmax(proba, axis=1)

    predictions = le.inverse_transform(pred_idx)
    confidence = np.max(proba, axis=1).round(2)

    return predictions, confidence


def GLORIA_v9(
    database_path: str,
    data_unknown_path: str,
    target_columns: list[str],
    optimize_hyperparams: bool = True,
    resample_method: str | None = None,
    n_iter_search: int = 25,
    cv_folds: int = 5,
    random_state: int = 42
):

    print("\n=========== GLORIA v9 ===========\n")

    models_dir = "GLORIA_models"
    os.makedirs(models_dir, exist_ok=True)

    output_path = data_unknown_path.replace(".xlsx", "_v9.xlsx")

    # === Load data ===
    data_known = pd.read_excel(database_path, sheet_name="Windows")
    data_unknown = pd.read_excel(data_unknown_path)

    # === Feature selection ===
    exclude_cols = target_columns + [
        'ID', 'Reference', 'ID(Ref)', 'Site', 'City', 'Country',
        'Region', 'Global region', 'Form', 'Data Method?', 'Colour',
        'Mg/Ca', 'Mg', 'Date - Early', 'Date - Mean',
        'Date - Late', 'Date',
        '(Na2O + MgO)/Sommes des autres'
    ]

    feature_columns = [c for c in data_known.columns if c not in exclude_cols]
    common_columns = [c for c in feature_columns if c in data_unknown.columns]

    replace_map = {'': 0, 'REF': 0, '#VALUE!': 0, '-': 0, '<LOD': 0, '< LOD': 0, np.nan: 0}

    X_known = (
        data_known[common_columns]
        .replace(replace_map)
        .infer_objects(copy=False)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    X_unknown = (
        data_unknown[common_columns]
        .replace(replace_map)
        .infer_objects(copy=False)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    # === Database hash ===
    db_hash = dataframe_hash(
        pd.concat([X_known, data_known[target_columns]], axis=1)
    )

    metadata_path = os.path.join(models_dir, "metadata.json")
    reuse_models = False

    if os.path.exists(metadata_path):
        metadata = load_json(metadata_path)
        if metadata.get("database_hash") == db_hash:
            reuse_models = input(
                " Database unchanged. Reuse models? [y/n]: "
            ).strip().lower() == "y"

    results, confidences = {}, {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    for target in target_columns:

        model_path = os.path.join(models_dir, f"{target}_model.joblib")

        if reuse_models and os.path.exists(model_path):
            print(f" Loading model for {target}")
            bundle = joblib.load(model_path)
        else:
            bundle = train_model_for_target(
                X=X_known,
                y=data_known[target],
                target_name=target,
                common_columns=common_columns,
                optimize_hyperparams=optimize_hyperparams,
                resample_method=resample_method,
                cv=cv,
                n_iter_search=n_iter_search,
                random_state=random_state
            )
            joblib.dump(bundle, model_path)

        preds, conf = predict_with_bundle(bundle, X_unknown)
        results[target] = preds
        confidences[f"{target}_confidence"] = conf

    if not reuse_models:
        save_json(
            metadata_path,
            {
                "database_hash": db_hash,
                "features": common_columns,
                "targets": target_columns,
                "created_on": datetime.now().isoformat()
            }
        )

    df_out = pd.DataFrame(results)
    for k, v in confidences.items():
        df_out[k] = v

    if "Samples" in data_unknown.columns:
        df_out["Samples"] = data_unknown["Samples"]

    df_out.to_excel(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

    return df_out

predictions = GLORIA_v9(
    database_path='GLORIA/Database.xlsx',
    data_unknown_path='GLORIA/Samples_to_train.xlsx',
    target_columns=['Century', 'Global region'],
    optimize_hyperparams=True,
    resample_method="smote",  # Options: None, "smote", "undersample"
    n_iter_search=30
)

#%%% GLORIA V10

CONFIG = {
    "random_state": 42,
    "cv_folds": 5,
    "scoring": "f1_weighted",

    # --- Resampling ---
    "smote_min_samples": 10,

    # --- Random Forest base ---
    "rf_base_params": {
        "class_weight": "balanced",
        "n_jobs": -1
    },

    # --- Hyperparameter search ---
    "rf_param_dist": {
        "n_estimators": np.arange(300, 501, 50),
        "max_depth": [30, 40, 50],
        "min_samples_split": np.arange(2, 11),
        "min_samples_leaf": [1],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False]
    },

    # --- Calibration ---
    "calibration": {
        "method": "sigmoid",
        "cv": 5
    }
}


def dataframe_hash(df: pd.DataFrame) -> str:
    """Hash only the meaningful content of a DataFrame."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def train_model_for_target(
    X: pd.DataFrame,
    y: pd.Series,
    target_name: str,
    common_columns: list[str],
    optimize_hyperparams: bool,
    resample_method: str | None,
    n_iter_search: int,
    config: dict
):
    print(f" Training model for {target_name}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    class_counts = np.bincount(y_encoded)
    min_samples = class_counts.min()

    X_train, y_train = X_scaled.copy(), y_encoded.copy()

    # --- Resampling guard ---
    if resample_method == "smote":
        if min_samples < config["smote_min_samples"]:
            print(f" SMOTE skipped (min class = {min_samples})")
        else:
            smote = SMOTE(random_state=config["random_state"])
            X_train, y_train = smote.fit_resample(X_train, y_train)

    elif resample_method == "undersample":
        rus = RandomUnderSampler(random_state=config["random_state"])
        X_train, y_train = rus.fit_resample(X_train, y_train)

    rf = RandomForestClassifier(
        random_state=config["random_state"],
        **config["rf_base_params"]
    )

    cv = StratifiedKFold(
        n_splits=config["cv_folds"],
        shuffle=True,
        random_state=config["random_state"]
    )

    # --- Hyperparameter tuning ---
    if optimize_hyperparams:
        search = RandomizedSearchCV(
            rf,
            param_distributions=config["rf_param_dist"],
            n_iter=n_iter_search,
            cv=cv,
            scoring=config["scoring"],
            random_state=config["random_state"],
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        rf = search.best_estimator_

        print(f"Best CV {config['scoring']}: {search.best_score_:.3f}")

    # --- Final fit ---
    rf.fit(X_train, y_train)

    # --- Probability calibration ---
    calibrated_rf = CalibratedClassifierCV(
        rf,
        method=config["calibration"]["method"],
        cv=config["calibration"]["cv"]
    )

    calibrated_rf.fit(X_train, y_train)

    print("✔ Probabilities calibrated")

    return {
        "model": calibrated_rf,
        "scaler": scaler,
        "label_encoder": le,
        "features": common_columns
    }


def compute_confusion_matrix(
    X: pd.DataFrame,
    y: pd.Series,
    target_name: str,
    common_columns: list[str],
    config: dict,
    optimize_hyperparams: bool = True,
    resample_method: str | None = None,
    n_iter_search: int = 25
):
    """
    Compute confusion matrix using stratified cross-validation
    """

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(
        n_splits=config["cv_folds"],
        shuffle=True,
        random_state=config["random_state"]
    )

    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in cv.split(X_scaled, y_encoded):

        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        # --- Optional resampling ---
        if resample_method == "smote":
            class_counts = np.bincount(y_train)
            if class_counts.min() >= config["smote_min_samples"]:
                smote = SMOTE(random_state=config["random_state"])
                X_train, y_train = smote.fit_resample(X_train, y_train)

        rf = RandomForestClassifier(
            random_state=config["random_state"],
            **config["rf_base_params"]
        )

        if optimize_hyperparams:
            search = RandomizedSearchCV(
                rf,
                param_distributions=config["rf_param_dist"],
                n_iter=n_iter_search,
                scoring=config["scoring"],
                cv=3,
                random_state=config["random_state"],
                n_jobs=-1
            )
            search.fit(X_train, y_train)
            rf = search.best_estimator_
        else:
            rf.fit(X_train, y_train)

        calibrated_rf = CalibratedClassifierCV(
            rf,
            method=config["calibration"]["method"],
            cv=3
        )
        calibrated_rf.fit(X_train, y_train)

        y_pred = calibrated_rf.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    cm = confusion_matrix(y_true_all, y_pred_all)

    labels = le.inverse_transform(np.arange(len(le.classes_)))

    return cm, labels

def plot_confusion_matrix(cm, labels, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.show()
    
    
def compute_label_instability(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
    resample_method: str | None = None,
):
    """
    Measure per-sample label instability using stratified CV.
    Returns a DataFrame with:
    - predicted class frequency
    - instability score
    - mean confidence
    """

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(
        n_splits=config["cv_folds"],
        shuffle=True,
        random_state=config["random_state"]
    )

    n_samples = len(y)
    n_classes = len(le.classes_)

    disagreement = np.zeros(n_samples)
    confidence_sum = np.zeros(n_samples)
    pred_counts = np.zeros((n_samples, n_classes))

    for train_idx, test_idx in cv.split(X_scaled, y_encoded):

        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        # --- Optional resampling ---
        if resample_method == "smote":
            class_counts = np.bincount(y_train)
            if class_counts.min() >= config["smote_min_samples"]:
                smote = SMOTE(random_state=config["random_state"])
                X_train, y_train = smote.fit_resample(X_train, y_train)

        rf = RandomForestClassifier(
            random_state=config["random_state"],
            **config["rf_base_params"]
        )

        rf.fit(X_train, y_train)

        calibrated_rf = CalibratedClassifierCV(
            rf,
            method=config["calibration"]["method"],
            cv=3
        )
        calibrated_rf.fit(X_train, y_train)

        proba = calibrated_rf.predict_proba(X_test)
        y_pred = np.argmax(proba, axis=1)
        conf = np.max(proba, axis=1)

        for i, idx in enumerate(test_idx):
            pred_counts[idx, y_pred[i]] += 1
            confidence_sum[idx] += conf[i]
            if y_pred[i] != y_encoded[idx]:
                disagreement[idx] += 1

    instability = disagreement / config["cv_folds"]
    mean_confidence = confidence_sum / config["cv_folds"]

    dominant_pred = pred_counts.argmax(axis=1)
    dominant_pred_label = le.inverse_transform(dominant_pred)

    df_diag = pd.DataFrame({
        "true_label": le.inverse_transform(y_encoded),
        "dominant_prediction": dominant_pred_label,
        "instability": instability.round(2),
        "mean_confidence": mean_confidence.round(2)
    })

    return df_diag

def predict_with_bundle(bundle, X_unknown):
    model = bundle["model"]
    scaler = bundle["scaler"]
    le = bundle["label_encoder"]
    features = bundle["features"]

    X_scaled = scaler.transform(X_unknown[features])

    proba = model.predict_proba(X_scaled)
    pred_idx = np.argmax(proba, axis=1)

    return (
        le.inverse_transform(pred_idx),
        np.max(proba, axis=1).round(2)
    )


def GLORIA_v10(
    database_path: str,
    data_unknown_path: str,
    target_columns: list[str],
    optimize_hyperparams: bool = True,
    resample_method: str | None = None,
    n_iter_search: int = 25
):
    """
    GLORIA v10
    - Centralized CONFIG
    - Model persistence
    - Optional retraining
    - Probability calibration
    """

    print("\n================ GLORIA v10 ================\n")

    # === Paths ===
    models_dir = "GLORIA_models"
    os.makedirs(models_dir, exist_ok=True)

    output_path = data_unknown_path.replace(".xlsx", "_v10.xlsx")

    # === Load data ===
    data_known = pd.read_excel(database_path, sheet_name="clean")
    data_unknown = pd.read_excel(data_unknown_path, sheet_name="withoutNaMg")

    # === Feature selection ===
    exclude_cols = target_columns + [
        'ID', 'Reference', 'ID(Ref)', 'Site', 'City', 'Country',
        'Region', 'Global region', 'Form', 'Data Method?', 'Colour',
        'Mg/Ca', 'Mg', 'Date - Early', 'Date - Mean',
        'Date - Late', 'Date',
        '(Na2O + MgO)/Sommes des autres'
    ]

    feature_columns = [c for c in data_known.columns if c not in exclude_cols]
    common_columns = [c for c in feature_columns if c in data_unknown.columns]

    # === Cleaning ===
    replace_map = {'': 0, 'REF': 0, '#VALUE!': 0, '-': 0, '<LOD': 0, '< LOD': 0, np.nan: 0, 'BD': 0, 'ND': 0}

    X_known = (
        data_known[common_columns]
        .replace(replace_map)
        .infer_objects(copy=False)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )
    
    # =====================================================
    # LABEL INSTABILITY DIAGNOSTIC (KNOWN DATA)
    # =====================================================
    print("\nRunning label instability diagnostic (Global region)...")
    
    diagnostic_region = compute_label_instability(
        X=X_known,
        y=data_known["Global region"],
        config=CONFIG,
        resample_method="smote"
    )
    
    diagnostic_region["ID"] = data_known["ID"]
    diagnostic_region["Global region"] = data_known["Global region"]
    
    ambiguous = diagnostic_region[diagnostic_region["instability"] > 0.6]

    # print("Ambiguous samples:")
    # print(ambiguous.sort_values("instability", ascending=False))
    
    # suspects = diagnostic_region[
    # (diagnostic_region["instability"] > 0.6) &
    # (diagnostic_region["true_label"] != diagnostic_region["dominant_prediction"])
    # ]
    # print(suspects)
    # print("Suspects:", (diagnostic_region["true_label"] != diagnostic_region["dominant_prediction"]).sum())
    
    misclassified = diagnostic_region[
    diagnostic_region["true_label"] != diagnostic_region["dominant_prediction"]
    ]
    print("Total misclassified:", len(misclassified))
    print(misclassified.sort_values("instability", ascending=False).head(10))

    # print("\nPotentially inconsistent samples:")
    # print(suspects.sort_values("instability", ascending=False).head(15))
    
    columns_to_export = [
    "ID", "true_label", "dominant_prediction", "instability", "mean_confidence"
    ]
    available_columns = [c for c in columns_to_export if c in misclassified.columns]

    misclassified_to_export = misclassified[available_columns]
    misclassified_to_export.to_excel(
    "GLORIA_misclassified_samples.xlsx",
    index=False
    )
    

    X_unknown = (
        data_unknown[common_columns]
        .replace(replace_map)
        .infer_objects(copy=False)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    results = {}
    confidences = {}

    # === Loop per target ===
    for target in target_columns:

        print(f"\n{'='*60}")
        print(f" Target: {target}")

        model_path = os.path.join(models_dir, f"{target}_model.joblib")

        # =====================================================
        # LOAD EXISTING MODEL
        # =====================================================
        if os.path.exists(model_path):

            answer = input(f"Reuse existing model for {target}? [y/n]: ").strip().lower()

            if answer == "y":
                print(" Loading model bundle")

                bundle = joblib.load(model_path)

            else:
                print(" Retraining requested")

                bundle = train_model_for_target(
                    X=X_known,
                    y=data_known[target],
                    target_name=target,
                    common_columns=common_columns,
                    optimize_hyperparams=optimize_hyperparams,
                    resample_method=resample_method,
                    n_iter_search=n_iter_search,
                    config=CONFIG
                )

                joblib.dump(bundle, model_path)

        # =====================================================
        # TRAIN NEW MODEL
        # =====================================================
        else:
            print(" No existing model found → training")

            bundle = train_model_for_target(
                X=X_known,
                y=data_known[target],
                target_name=target,
                common_columns=common_columns,
                optimize_hyperparams=optimize_hyperparams,
                resample_method=resample_method,
                n_iter_search=n_iter_search,
                config=CONFIG
            )

            joblib.dump(bundle, model_path)

        # =====================================================
        # PREDICTION
        # =====================================================
        preds, confs = predict_with_bundle(bundle, X_unknown)

        results[target] = preds
        confidences[f"{target}_confidence"] = confs

    # === Export ===
    df_out = pd.DataFrame(results)

    for k, v in confidences.items():
        df_out[k] = v

    if "Samples" in data_unknown.columns:
        df_out["Samples"] = data_unknown["Samples"]

    df_out.to_excel(output_path, index=False)

    print(f"\n Predictions saved to: {output_path}")
    print(f" Models stored in: {models_dir}")

    
    # === Century confusion matrix ===
    cm_century, labels_century = compute_confusion_matrix(
        X=X_known,
        y=data_known["Century"],
        target_name="Century",
        common_columns=common_columns,
        config=CONFIG,
        optimize_hyperparams=True,
        resample_method="smote",
        n_iter_search=20
    )
    
    plot_confusion_matrix(
        cm_century,
        labels_century,
        title="Confusion matrix – Chronological attribution (Century)"
    )
    
    # === Global region confusion matrix ===
    cm_region, labels_region = compute_confusion_matrix(
        X=X_known,
        y=data_known["Global region"],
        target_name="Global region",
        common_columns=common_columns,
        config=CONFIG,
        optimize_hyperparams=True,
        resample_method="smote",
        n_iter_search=20
    )
    
    plot_confusion_matrix(
        cm_region,
        labels_region,
        title="Confusion matrix – Geographical attribution (Global region)"
    )
    
    np.savez(
    "confusion_matrices_GLORIA_v10.npz",
    cm_century=cm_century,
    labels_century=labels_century,
    cm_region=cm_region,
    labels_region=labels_region
)

    return df_out

predictions = GLORIA_v10(
    database_path='Database_windows_only.xlsx',
    data_unknown_path='GLORIA_Ste_Chapelle.xlsx',
    target_columns=['Century', 'Global region'],
    optimize_hyperparams=True,
    resample_method="smote",  # Options: None, "smote", "undersample"
    n_iter_search=30
)