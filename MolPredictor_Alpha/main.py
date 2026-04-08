import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from data_acquisition import check_data_exist,data_loading,data_downloading
from data_preprocessing import show_molecule
from KNN_train import KNN_train
from Logistic_Regression_train import LR_train,LR_train_L1,LR_train_L2
from RF_train import RF_train
from XGboost_train import XGboost_train


def main():
    # 1.Main Entry Point: Orchestrate the Tox21 dataset preparation and loading process.
    final_path = None
    answer = input("Do you have the tox21.csv in you computer?(y/n)").strip().lower()
    if answer == "y":
        final_path = check_data_exist()
    elif answer == "n":
        print("Starting download process...")
        final_path = data_downloading()
    else:
        print("Invalid input, please enter 'y' or 'n'.")

    # 2.This code block is used to load the Tox21 toxicity prediction dataset.
    if final_path:
        tox21_data = data_loading(final_path)

        if tox21_data is not None:
            print("\n" + "="*30)
            print("Tox21 Dataset Preview:")
            # use head() preview the first 5 rows
            print(tox21_data.head())
            
            # Check the number of missing values (NaN) in the dataset.
            print("\nMissing Values Count (per task):")
            print(tox21_data.isnull().sum().head(10)) # only check for the first 10 columns
            print("="*30)
    else:
        print("Process terminated: No valid data path.")

    # 3.Prompt user for an index and visualize the corresponding molecule from the Tox21 dataset.
    df = tox21_data
    idx = input("You want to check which number in tox21:")
    idx = int(idx)
    show_molecule(df,idx)

    # 4.show the result of KNN
    # KNN_train(df)       # now test best is random_state = 0 ,n_neighbors= 7

    # 5.show the result of Logistic Regression
    # LR_train(df)

    # 6.show the result of Logistic Regression(L1 regularization)
    # LR_train_L1(df)

    # 7.show the result of Logistic Regression(L2 regularization)
    # LR_train_L2(df)

    # 8.show the result of Random Forest
    # RF_train(df)

    # 9.
    XGboost_train(df)

if __name__ == '__main__':
    main()


