import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.model_selection import RandomizedSearchCV
from data_preprocessing import extract_features

def XGboost_train(df,use_gpu=True):
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
    

    # 1. Calculate dynamic positive class weight
    n_pos = np.sum(y_train)
    n_neg = len(y_train) - n_pos
    pos_weight = n_neg / n_pos
    
    # 2. Define grid search space (balancing mathematical pruning and generalization)
    param_grid = {
        'max_depth': [3, 5, 7],     
        'min_child_weight': [1, 3],        
        'gamma': [0, 0.1],          
        'learning_rate': [0.03, 0.05],
        'n_estimators': [300],
        'subsample': [0.8],     
        'colsample_bytree': [0.8], 
        'reg_alpha': [0.01, 0.1],        
        'reg_lambda': [1, 5],
        # 调回理性权重，防止过拟合
        'scale_pos_weight': [round(pos_weight, 2), round(pos_weight * 1.5, 2)] 
    }
    # 3. Initialize model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        n_estimators=300,                      
        random_state=42,
        eval_metric='auc'
    )

    # 4. Cross-validation settings
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # 5. Perform search
    print(f"Launch the XGBoost strategy engine...")

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,  
        n_iter=30,                  
        scoring='roc_auc',
        cv=skf,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(x_train, y_train)

    best_model = random_search.best_estimator_

    # Calculate metrics
    train_auc = roc_auc_score(y_train, best_model.predict_proba(x_train)[:, 1])
    test_auc = roc_auc_score(y_test, best_model.predict_proba(x_test)[:, 1])
    threshold = 0.3  # 只要有 30% 概率是毒性，就报警！
    y_test_proba = best_model.predict_proba(x_test)[:, 1]
    y_test_pred = (y_test_proba >= threshold).astype(int)
    
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Extract feature importance (XGBoost gain)
    importance_dict = best_model.get_booster().get_score(importance_type='gain')
    n_features = x_train.shape[1]   # total number of features (should be 1024)
    importances = np.zeros(n_features)
    for k, v in importance_dict.items():
    # key format is 'f123', extract numeric part as index
        idx = int(k[1:])
        importances[idx] = v
    # Optional: normalize so that sum of importances equals 1
    if importances.sum() > 0:
      importances = importances / importances.sum()
    # Sort and take top 5
    top_indices = np.argsort(importances)[-5:][::-1]
    top_bits = {f"Bit {i}": round(importances[i], 4) for i in top_indices}

    # Construct report dictionary
    report = {
        'train_auc': train_auc,
        'test_auc': test_auc,
        'best_params': random_search.best_params_,
        'cm': cm,               # confusion matrix
        'y_test': y_test,       # added for classification_report
        'y_test_pred': y_test_pred, # added
        'top_bits': top_bits    # originally top_5_bits
    }
    print_XGboost_experiment_report(report)


def print_XGboost_experiment_report(report):
    # Overfitting check
    print("\n-- Overfitting Check --")
    print(f"Train AUC: {report['train_auc']:.4f}")
    print(f"Test AUC: {report['test_auc']:.4f}")
    print(f"Gap: {report['train_auc'] - report['test_auc']:.4f}")

    # Separator and title
    print("\n===========================")
    print("Project MolPredictor_Alpha: XGBoost Report(Automatic hyperparameter tuning results)")
    print("Task: NR-AR")
    print("===========================")

    # ROC-AUC
    print(f"ROC-AUC Score: {report['test_auc']:.4f}\n")

    # Confusion matrix (format consistent with Random Forest)
    cm = report['cm']
    print("Confusion Matrix:")
    # Manually format, right-aligned with consistent width (adjust based on actual digit length)
    print(f"[{cm[0][0]:4d}  {cm[0][1]:4d}]")
    print(f" [{cm[1][0]:4d}  {cm[1][1]:4d}]")

    # Detailed classification report
    print("\nDetailed Metrics:")
    print(classification_report(report['y_test'], report['y_test_pred']))

    # Important features
    print("\n--- Top 5 Important Bits (XGBoost Gain) ---")
    for bit, val in report['top_bits'].items():
        print(f"{bit}: {val:.4f}")
