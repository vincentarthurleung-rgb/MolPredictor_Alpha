import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from data_preprocessing import extract_features


def KNN_train(df):
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

    # 2.train the model
    estimator = KNeighborsClassifier(n_neighbors=7,weights='distance', n_jobs=-1)
    estimator.fit(x_train,y_train)

    # 3.Model evaluation
    print(f'accuracy:{estimator.score(x_test,y_test)}')

    # Predict class (0 or 1)
    y_pred = estimator.predict(x_test)
    # Predict probabilities (used for calculating AUC)
    y_prob = estimator.predict_proba(x_test)[:, 1]

    print(f"Project MolPredictor_Alpha: KNN Report")
    print(f"Task: {task_name}")
    print("="*30)
    print(f"ROC-AUC score: {roc_auc_score(y_test, y_prob):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nDetailed metrics:")
    print(classification_report(y_test, y_pred))

