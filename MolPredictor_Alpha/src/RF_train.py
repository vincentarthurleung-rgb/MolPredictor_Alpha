import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

from data_preprocessing import extract_features


def RF_train(df):
    """
    Train a Random Forest model for a single toxicity task from the Tox21 dataset,
    evaluate performance, check for overfitting, analyze feature importance, and
    call an automatic tuning function that balances AUC and Gap.

    Parameters:
        df (pd.DataFrame): Tox21 DataFrame containing SMILES column and multi-label columns.

    Workflow:
        1. Data preprocessing:
           1.1 Call extract_features() to obtain feature matrix X (1024-bit Morgan fingerprints) and labels y.
           1.2 Set the target task (default task_idx=0, i.e., 'NR-AR' toxicity task).
           1.3 Remove NaN values from the label column, keeping only valid samples.
           1.4 Split the dataset into 80% training / 20% testing (stratified split, fixed random seed).
        2. Create a Random Forest model (n_estimators=200, max_depth=8, min_samples_leaf=4, class_weight='balanced').
        3. Train the model and check for overfitting:
           - Compute AUC on training and test sets.
           - Print Train AUC, Test AUC, and Gap (if Gap > 0.1, it indicates severe overfitting).
        4. Model prediction: Obtain predicted classes and probabilities on the test set.
        5. Model evaluation: Output ROC-AUC, confusion matrix, and classification report (precision/recall/F1).
        6. Feature importance analysis: Based on Gini importance, sort and output the top 5 most important fingerprint bits.
        7. Automatic hyperparameter tuning: Call RF_optimize_with_auc_gap_tradeoff() to search for optimal parameters balancing AUC and Gap,
           and print the report under the best parameters.

    Returns:
        None (prints evaluation results, important features, and tuning report directly).
    """
    # 1.Preprocessing of data
    X,y = extract_features(df) # X is feature,y is label
        # 1.1 set a goal
    task_idx = 0
    task_name = 'NR-AR'
        # 1.2 let nan values get out
    y_task = y.iloc[:,task_idx].values
    mask = ~np.isnan(y_task) # Find the indices of non-null values.

    X_final = X[mask]
    y_final = y_task[mask]

        # 1.3 Split the dataset (80% training, 20% testing).
    x_train,x_test,y_train,y_test = train_test_split(X_final, y_final, test_size = 0.2, random_state=0, stratify=y_final)
    

    # 2.Create a random forest model.
    estimator = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=4,
        class_weight='balanced',
        n_jobs=-1,
        random_state=23
    )

    # 3.train the model
    estimator.fit(x_train, y_train)

        # 3.1 Check for overfitting.
    train_probs = estimator.predict_proba(x_train)[:, 1]
    test_probs = estimator.predict_proba(x_test)[:, 1]

    train_auc = roc_auc_score(y_train, train_probs)
    test_auc = roc_auc_score(y_test, test_probs)

    print(f"\n--- Overfitting Check ---")
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Gap: {train_auc - test_auc:.4f}") # If the gap > 0.1, it indicates severe overfitting.


    # 4.Model prediction
    y_pre = estimator.predict(x_test)
    y_prob = estimator.predict_proba(x_test)[:, 1]

    # 5.Model evaluation
    print("\n" + "="*30)
    print(f"Project MolPredictor_Alpha: Random Forest Report")
    print(f"Task: {task_name}")
    print("="*30)

        # Key metrics
    auc_score = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc_score:.4f}")
    
        # Confusion matrix: see how many of those 62 toxic molecules were captured.
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pre))
    
    print("\nDetailed Metrics:")
    print(classification_report(y_test, y_pre))

    # 6.Get feature importance.
    importances = estimator.feature_importances_
        # Sort and take the top 5.
    indices = np.argsort(importances)[::-1]

    print("\n--- Top 5 Important Bits (RF Gini Importance) ---")
    for i in range(5):
        print(f"Bit {indices[i]}: {importances[indices[i]]:.4f}")

    # 7.Automatically tune hyperparameters, select the best, balancing AUC and Gap, and print the report under the best parameters.
    RF_optimize_with_auc_gap_tradeoff_02(x_train, y_train,x_test,y_test,task_name)


def RF_optimize_with_auc_gap_tradeoff_01(x_train, y_train,x_test,y_test,task_name):
    """
    Perform hyperparameter search for Random Forest using a custom balanced score
    (Test AUC - 0.5 * Gap), select the best parameters, train the final model,
    and output a detailed evaluation report (including overfitting check and feature importance).

    Parameters:
        x_train, y_train : Training features and labels
        x_test, y_test   : Test features and labels
        task_name        : Name of the current toxicity task (used for printing reports)

    Workflow:
        1. Define the search space (max_depth, min_samples_leaf, n_estimators).
        2. Manual triple loop + 5-fold cross-validation for each parameter combination:
           - Compute mean validation AUC (mean_test)
           - Compute mean gap between training AUC and validation AUC (mean_gap)
           - Balanced score = mean_test - 0.5 * mean_gap
        3. Select the parameter combination with the highest balanced score.
        4. Create and train the final Random Forest model using the best parameters.
        5. Check for overfitting (compute train/test AUC and Gap).
        6. Output predictions: ROC-AUC, confusion matrix, classification report.
        7. Output feature importance (Top 5 fingerprint bits).

    Returns:
        None (prints tuning results, final evaluation report, and important features).
    """
    # (original function body)
    best_score = -np.inf
    best_params = None
    
    # 1. Define the search space.
    if hasattr(x_train, 'values'):
        x_train = x_train.values
        y_train = y_train.values
    if hasattr(x_test, 'values'):
        x_test = x_test.values

    param_grid = {
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'n_estimators': [200, 500, 1000]
    }
    # 2. Manually write a loop to implement the "trade-off" logic.
    print("\n--- Start automatic hyperparameter tuning (based on 5-fold CV on the training set)---")
    for d in param_grid['max_depth']:
        for l in param_grid['min_samples_leaf']:
            for n in param_grid['n_estimators']:
                rf = RandomForestClassifier(max_depth=d, min_samples_leaf=l, n_estimators=n, 
                                            class_weight='balanced', n_jobs=-1, random_state=23)
                
                # Cross-validation
                kf = KFold(n_splits=5, shuffle=True, random_state=23)
                test_aucs, train_aucs = [], []
                
                for train_idx, val_idx in kf.split(x_train):
                    rf.fit(x_train[train_idx], y_train[train_idx])
                    # Calculate the AUC for training and validation.
                    train_aucs.append(roc_auc_score(y_train[train_idx], rf.predict_proba(x_train[train_idx])[:,1]))
                    test_aucs.append(roc_auc_score(y_train[val_idx], rf.predict_proba(x_train[val_idx])[:,1]))
                
                mean_test = np.mean(test_aucs)
                mean_gap = np.mean(train_aucs) - mean_test
                
                # [Core modification]：Balanced score = Test AUC - 0.5 * Gap
                custom_score = mean_test - 0.5 * mean_gap
                
                if custom_score > best_score:
                    best_score = custom_score
                    best_params = {'max_depth': d, 'min_samples_leaf': l, 'n_estimators': n}

    if best_params is None:
        print("Error: No valid parameter combination found. Using default parameters.")
        best_params = {'max_depth': 8, 'min_samples_leaf': 4, 'n_estimators': 200}
    
    print(f"Optimal parameters balancing both AUC and Gap: {best_params}")

    
    # 3. Create the final model using the best parameters.
    estimator = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_leaf=best_params['min_samples_leaf'],
        class_weight='balanced',
        n_jobs=-1,
        random_state=23
    )
    
    # 4. Train the final model.
    estimator.fit(x_train, y_train)
    
        # 4.1 Check for overfitting.
    train_probs = estimator.predict_proba(x_train)[:, 1]
    test_probs = estimator.predict_proba(x_test)[:, 1]

    train_auc = roc_auc_score(y_train, train_probs)
    test_auc = roc_auc_score(y_test, test_probs)

    print(f"\n--- Overfitting Check ---")
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Gap: {train_auc - test_auc:.4f}") # If the gap > 0.1, it indicates severe overfitting.


    # 4.Model prediction
    y_pre = estimator.predict(x_test)
    y_prob = estimator.predict_proba(x_test)[:, 1]

    # 5.Model evaluation
    print("\n" + "="*30)
    print(f"Project MolPredictor_Alpha: Random Forest Report(Automatic hyperparameter tuning results)")
    print(f"Task: {task_name}")
    print("="*30)

        # Key metrics
    auc_score = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc_score:.4f}")
    
        # Confusion matrix: see how many of those 62 toxic molecules were captured.
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pre))
    
    print("\nDetailed Metrics:")
    print(classification_report(y_test, y_pre))

    # 6.Get feature importance.
    importances = estimator.feature_importances_
        
        # Sort and take the top 5.
    indices = np.argsort(importances)[::-1]

    print("\n--- Top 5 Important Bits (RF Gini Importance) ---")
    for i in range(5):
        print(f"Bit {indices[i]}: {importances[indices[i]]:.4f}")


def RF_optimize_with_auc_gap_tradeoff_02(x_train, y_train,x_test,y_test,task_name):
    """
    Perform Random Forest hyperparameter tuning using a custom balanced score:
    (mean validation AUC - penalty_coefficient * |actual_gap - target_gap|).
    Select the best parameters, train the final model, and output a detailed evaluation report
    (including overfitting check and feature importance).

    Parameters:
        x_train, y_train : Training features and labels (numpy array or DataFrame)
        x_test, y_test   : Test features and labels
        task_name        : Name of the current toxicity task (used for report title)

    Core idea:
        - Target Gap is set to 0.1 (acceptable overfitting threshold).
        - Penalty coefficient is set to 2.0, forcing the tuning to drive Gap towards 0.1.
        - Final score = mean validation AUC - penalty_coefficient x |actual_gap - 0.1|.
        - Higher score indicates a better balance between high AUC and low overfitting.

    Workflow:
        1. Convert inputs to numpy arrays (for easy indexing).
        2. Define the hyperparameter search space (max_depth, min_samples_leaf, n_estimators).
        3. Manual triple loop + 5-fold cross-validation for each parameter combination:
           - Compute mean validation AUC (mean_test)
           - Compute mean gap between training AUC and validation AUC (mean_gap)
           - Compute deviation from target Gap = |mean_gap - 0.1|
           - Balanced score = mean_test - 2.0 × deviation
        4. Select the parameter combination with the highest balanced score.
        5. Create and train the final Random Forest model using the best parameters.
        6. Check for overfitting (print train/test AUC and Gap).
        7. Output predictions: ROC-AUC, confusion matrix, classification report.
        8. Output feature importance (Top 5 fingerprint bits).

    Returns:
        None (prints tuning results, final evaluation report, and important features).
    """
    # (original function body)
    # (original function body)
    best_score = -np.inf
    best_params = None
    
    # 1. Define the search space.
    if hasattr(x_train, 'values'):
        x_train = x_train.values
        y_train = y_train.values
    if hasattr(x_test, 'values'):
        x_test = x_test.values

    param_grid = {
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'n_estimators': [200, 500, 1000]
    }
    # 2. Manually write a loop to implement the "trade-off" logic.
    print("\n--- Start automatic hyperparameter tuning (based on 5-fold CV on the training set)---")
    for d in param_grid['max_depth']:
        for l in param_grid['min_samples_leaf']:
            for n in param_grid['n_estimators']:
                rf = RandomForestClassifier(max_depth=d, min_samples_leaf=l, n_estimators=n, 
                                            class_weight='balanced', n_jobs=-1, random_state=23)
                
                # Cross-validation
                kf = KFold(n_splits=5, shuffle=True, random_state=23)
                test_aucs, train_aucs = [], []
                
                for train_idx, val_idx in kf.split(x_train):
                    rf.fit(x_train[train_idx], y_train[train_idx])
                    # Calculate the AUC for training and validation.
                    train_aucs.append(roc_auc_score(y_train[train_idx], rf.predict_proba(x_train[train_idx])[:,1]))
                    test_aucs.append(roc_auc_score(y_train[val_idx], rf.predict_proba(x_train[val_idx])[:,1]))
                
                mean_test = np.mean(test_aucs)
                mean_gap = np.mean(train_aucs) - mean_test
                
                target_gap = 0.1  # My target Gap value."
                gap_penalty_weight = 2.0  # Penalty coefficient: the larger it is, the more the Gap is forced to approach 0.1.
                
                # Calculate the current deviation
                gap_deviation = abs(mean_gap - target_gap)
                
                # Final score = Mean validation AUC - Penalty for deviation from the target.
                custom_score = mean_test - (gap_penalty_weight * gap_deviation)
                
                if custom_score > best_score:
                    best_score = custom_score
                    best_params = {'max_depth': d, 'min_samples_leaf': l, 'n_estimators': n}

    if best_params is None:
        print("Error: No valid parameter combination found. Using default parameters.")
        best_params = {'max_depth': 8, 'min_samples_leaf': 4, 'n_estimators': 200}
    
    print(f"Optimal parameters balancing both AUC and Gap: {best_params}")

    
    # 3. Create the final model using the best parameters.
    estimator = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_leaf=best_params['min_samples_leaf'],
        class_weight='balanced',
        n_jobs=-1,
        random_state=23
    )
    
    # 4. Train the final model.
    estimator.fit(x_train, y_train)
    
        # 4.1 Check for overfitting.
    train_probs = estimator.predict_proba(x_train)[:, 1]
    test_probs = estimator.predict_proba(x_test)[:, 1]

    train_auc = roc_auc_score(y_train, train_probs)
    test_auc = roc_auc_score(y_test, test_probs)

    print(f"\n--- Overfitting Check ---")
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Gap: {train_auc - test_auc:.4f}") # If the gap > 0.1, it indicates severe overfitting.


    # 4.Model prediction
    y_pre = estimator.predict(x_test)
    y_prob = estimator.predict_proba(x_test)[:, 1]

    # 5.Model evaluation
    print("\n" + "="*30)
    print(f"Project MolPredictor_Alpha: Random Forest Report(Automatic hyperparameter tuning results)")
    print(f"Task: {task_name}")
    print("="*30)

        # Key metrics
    auc_score = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc_score:.4f}")
    
        # Confusion matrix: see how many of those 62 toxic molecules were captured.
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pre))
    
    print("\nDetailed Metrics:")
    print(classification_report(y_test, y_pre))

    # 6.Get feature importance.
    importances = estimator.feature_importances_
        
        # Sort and take the top 5.
    indices = np.argsort(importances)[::-1]

    print("\n--- Top 5 Important Bits (RF Gini Importance) ---")
    for i in range(5):
        print(f"Bit {indices[i]}: {importances[indices[i]]:.4f}")
