import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from data_preprocessing import extract_features

def LR_train(df):
    """
    Train a logistic regression model (L2 regularization) for a single toxicity task from the Tox21 dataset,
    evaluate performance, and analyze feature contributions.

    Parameters:
        df (pd.DataFrame): Tox21 DataFrame containing SMILES column and multi-label columns.

    Workflow:
        1. Data preprocessing:
           1.1 Call extract_features() to obtain feature matrix X (1024-bit Morgan fingerprints) and labels y.
           1.2 Set the target task (default task_idx=0, i.e., 'NR-AR' toxicity task).
           1.3 Remove NaN values from the label column, keeping only valid samples.
           1.4 Split the dataset into 80% training / 20% testing (stratified split).
        2. Feature engineering: Standardize the feature matrix using StandardScaler (L1/L2 regularization is scale-sensitive).
        3. Model training: Create a LogisticRegression object (solver='liblinear') and train on the standardized training set.
        4. Model evaluation and prediction:
           - Predict class labels and probabilities on the test set.
           - Output ROC-AUC, confusion matrix, and detailed metrics (precision, recall, F1).
        5. Feature contribution analysis: Extract the 1024 feature weights and identify the top 5 features with the largest weights (positive weights indicate stronger toxicity contribution).

    Returns:
        None (prints evaluation results and important features directly).
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
    
    # 2. Feature Engineering: Standardization
        # 2.1 Create a standardization object
    scaler = StandardScaler()
        # 2.2 Standardize the training set
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)  

    # 3.Model training
        # 3.1 Create a model object
    estimator = LogisticRegression(solver='liblinear', max_iter=500)
        # 3.2 train model
    estimator.fit(x_train_scaled, y_train)

    # 4. Model evaluation and prediction
    y_pre = estimator.predict(x_test_scaled)
    y_prob = estimator.predict_proba(x_test_scaled)[:, 1]
    print("\n" + "="*30)
    print(f"Project MolPredictor_Alpha: Logistic Regression Report")
    print(f"Task: {task_name}")
    print("="*30)
    # Core metrics
    auc_score = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc_score:.4f}")
    
    # Confusion matrix (to see how many toxic molecules were captured)
    print(confusion_matrix(y_test, y_pre))
    
    # Detailed metrics (Precision, Recall, F1)
    print("\n--- Detailed Metrics ---")
    print(classification_report(y_test, y_pre))

    # 5. Feature Contribution Analysis (Feature Importance)
        # Get the weights of the 1024 features
    weights = estimator.coef_[0]
    
        # Find the top 5 features with the largest weights (the larger the positive contribution, the stronger the toxicity)
    top_indices = np.argsort(weights)[-5:]

    print("\n--- Top 5 Toxic Structural Fragments (Indices) ---")
    for idx in reversed(top_indices):
        print(f"Fingerprint Bit {idx}: Weight {weights[idx]:.4f}")

def LR_train_L1(df):
    """
    Train a logistic regression model with L1 regularization (Lasso) for a single toxicity task from the Tox21 dataset,
    evaluate performance, and perform feature selection.

    Parameters:
        df (pd.DataFrame): Tox21 DataFrame containing SMILES column and multi-label columns.

    Workflow:
        1. Data preprocessing:
           1.1 Call extract_features() to obtain feature matrix X (1024-bit Morgan fingerprints) and labels y.
           1.2 Set the target task (default task_idx=0, i.e., 'NR-AR' toxicity task).
           1.3 Remove NaN values from the label column, keeping only valid samples.
           1.4 Split the dataset into 80% training / 20% testing (stratified split, fixed random seed).
        2. Feature standardization: Standardize the feature matrix using StandardScaler (L1 regularization is scale-sensitive).
        3. Enable L1 regularization:
           - Use solver='saga' (supports L1 regularization).
           - Set l1_ratio=1.0 for pure L1 regularization (Lasso).
           - C=0.1 increases the penalty strength, driving more feature weights to zero.
           - class_weight='balanced' automatically handles class imbalance.
        4. Model evaluation and prediction:
           - Predict class labels and probabilities on the test set.
           - Output ROC-AUC, confusion matrix, and detailed metrics (precision, recall, F1).
        5. Feature contribution analysis and L1 compression effect:
           - Extract the 1024 feature weights.
           - Identify the top 5 features with the largest weights (positive weights indicate stronger toxicity contribution).
           - Count the number of non-zero weights to demonstrate the sparsity effect of L1 regularization.

    Returns:
        None (prints evaluation results, important features, and compression effect directly).
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

    # 2.Standardization (L1 is very sensitive to scale, must be retained)
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    # 3. Enable L1 regularization
    # The smaller C, the stronger the penalty, and the more features become zero
    estimator = LogisticRegression( 
        solver='saga',
        l1_ratio=1.0,
        C=0.05,
        max_iter=5000,   
        random_state=23,
        class_weight='balanced'
    )

    estimator.fit(x_train_s, y_train)

    # 4. Model evaluation and prediction
    y_pre = estimator.predict(x_test_s)
    y_prob = estimator.predict_proba(x_test_s)[:, 1]
    print("\n" + "="*30)
    print(f"Project MolPredictor_Alpha: Logistic Regression Report(L1 regularization)")
    print(f"Task: {task_name}")
    print("="*30)
        # Core metrics
    auc_score = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc_score:.4f}")
    
        # Confusion matrix (to see how many toxic molecules were captured)
    print(confusion_matrix(y_test, y_pre))
    
        # Detailed metrics (Precision, Recall, F1)
    print("\n--- Detailed Metrics ---")
    print(classification_report(y_test, y_pre))

    # 5. Feature Contribution Analysis (Feature Importance)
        # Get the weights of the 1024 features
    weights = estimator.coef_[0]
    
        # Find the top 5 features with the largest weights (the larger the positive contribution, the stronger the toxicity)
    top_indices = np.argsort(weights)[-5:]

    print("\n--- Top 5 Toxic Structural Fragments (Indices) ---")
    for idx in reversed(top_indices):
        print(f"Fingerprint Bit {idx}: Weight {weights[idx]:.4f}")
        # Count the number of non-zero weights
    non_zero_count = np.sum(estimator.coef_ != 0)
    print(f"\nL1 compression effect: Only {non_zero_count} effective features retained out of 1024 features")

def LR_train_L2(df):
    """
    Train a logistic regression model with L2 regularization (Ridge) for a single toxicity task from the Tox21 dataset,
    evaluate performance, and analyze feature contributions.

    Parameters:
        df (pd.DataFrame): Tox21 DataFrame containing SMILES column and multi-label columns.

    Workflow:
        1. Data preprocessing:
           1.1 Call extract_features() to obtain feature matrix X (1024-bit Morgan fingerprints) and labels y.
           1.2 Set the target task (default task_idx=0, i.e., 'NR-AR' toxicity task).
           1.3 Remove NaN values from the label column, keeping only valid samples.
           1.4 Split the dataset into 80% training / 20% testing (stratified split, fixed random seed).
        2. Feature standardization: Standardize the feature matrix using StandardScaler (regularization is scale-sensitive).
        3. Enable L2 regularization (Ridge):
           - Set l1_ratio=0 for pure L2 regularization.
           - Use solver='lbfgs' (default efficient solver for L2 regularization).
           - C=0.01 sets the regularization strength (consistent with L1 version for fair comparison).
           - class_weight='balanced' automatically handles class imbalance.
        4. Model evaluation and prediction:
           - Predict class labels and probabilities on the test set.
           - Output ROC-AUC, confusion matrix, and detailed metrics (precision, recall, F1).
        5. Feature contribution analysis:
           - Extract the 1024 feature weights.
           - Identify the top 5 features with the largest weights (positive weights indicate stronger toxicity contribution).
           - Count the number of non-zero weights (L2 regularization does not force weights to exactly zero; this shows retained effective features).

    Returns:
        None (prints evaluation results and important features directly).
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

    # 2.Standardization (L2 is very sensitive to scale, must be retained)
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    # 3. Enable L2 regularization
    # The smaller C, the stronger the penalty, and the more features become zero
    estimator = LogisticRegression( 
        l1_ratio=0,           # 切换为 L2
        solver='lbfgs',         # L2 默认最稳、最快的求解器
        C=0.0001,                  # 同样设置正则化强度，方便横向对比
        class_weight='balanced', # 保持类别平衡一致
        max_iter=1000,
        random_state=23
    )

    estimator.fit(x_train_s, y_train)

    # 4. Model evaluation and prediction
    y_pre = estimator.predict(x_test_s)
    y_prob = estimator.predict_proba(x_test_s)[:, 1]
    print("\n" + "="*30)
    print(f"Project MolPredictor_Alpha: Logistic Regression Report(L2 regularization)")
    print(f"Task: {task_name}")
    print("="*30)
        # Core metrics
    auc_score = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc_score:.4f}")
    
        # Confusion matrix (to see how many toxic molecules were captured)
    print(confusion_matrix(y_test, y_pre))
    
        # Detailed metrics (Precision, Recall, F1)
    print("\n--- Detailed Metrics ---")
    print(classification_report(y_test, y_pre))

    # 5. Feature Contribution Analysis (Feature Importance)
        # Get the weights of the 1024 features
    weights = estimator.coef_[0]
    
        # Find the top 5 features with the largest weights (the larger the positive contribution, the stronger the toxicity)
    top_indices = np.argsort(weights)[-5:]

    print("\n--- Top 5 Toxic Structural Fragments (Indices) ---")
    for idx in reversed(top_indices):
        print(f"Fingerprint Bit {idx}: Weight {weights[idx]:.4f}")
        # Count the number of non-zero weights
    non_zero_count = np.sum(estimator.coef_ != 0)
    print(f"L2 regularization: All {weights.shape[0]} features have non-zero weights (no feature selection).")

