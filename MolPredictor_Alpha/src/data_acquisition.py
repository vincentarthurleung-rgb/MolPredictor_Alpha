import os
import requests
import pandas as pd
from pathlib import Path

# Interactively check if the Tox21 dataset exists.
def check_data_exist():
    """
    Interactively check if the Tox21 dataset exists.
    This function will continuously prompt the user to input a path until the standard 'tox21.csv' file is found in the specified directory.
    This validation mechanism ensures that the training data for the subsequent MolPredictor model is truly available.
    Returns:
        str: A validated folder path string containing the dataset file.
    """
    while True:
        # get the path from users
        data_directory = input('Please enter the folder where is the data, tox21, located (use a relative path):') or "data/raw"
        
        # Path about the file
        file_path = Path(data_directory)/"tox21.csv"

        # Judge the file exist or not
        if file_path.exists():
            print(f'The file has been successfully extracted : {file_path.name}')
            return data_directory
        else:
            print('Failed to retrieve the file. Please try again.')

# Automatically retrieve the Tox21 dataset from cloud storage.
def data_downloading():
    """
    Automatically retrieve the Tox21 dataset from the MoleculeNet database.
    
    Returns:
        str: Absolute or relative path to the directory for data storage.
    """
    # Define the download URL which is the official MoleculeNet source for Tox21 dataset
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"

    # Unified path for directory creation and file storage
    save_dir = Path("data/raw")
    save_path = save_dir/"tox21.csv.gz"
    save_dir.mkdir(parents=True, exist_ok=True)

    # downloading
    print("Fetching dataset from cloud storage...")
    
    try:
        # sending request
        response = requests.get(url, stream=True)
        response.raise_for_status() # if 404 or 500 give the error
        
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Download completed! File saved at: {save_path}")
        print("Note:This is a compressed archive (.gz),Pandas' read_csv can read it directly without manual decompression")
        
        return str(save_dir)
    
    except requests.exceptions.RequestException as e:
        # Catch network-related errors and display a hint.
        print(f"Network error occurred during download: {e}")
        return None

# Load the verified dataset from local disk into a DataFrame.
def data_loading(data_dir):
    """
    Load the Tox21 dataset into a Pandas DataFrame.

    This function automatically identifies the dataset file within the specified 
    directory, supporting both raw '.csv' and compressed '.csv.gz' formats 
    through pattern matching.

    Args:
        data_dir (str or Path): The directory path where the dataset is stored.

    Returns:
        pd.DataFrame: The loaded dataset if successful; None if no matching file is found.

    Raises:
        StopIteration: Handled internally if no file matching 'tox21.csv*' exists.
    """
    p = Path(data_dir)
    try:
        # Read the dataset and obtain the source data.
        file_path = next(p.glob('tox21.csv*'))
        tox21_data = pd.read_csv(file_path)
        print(f"Data loaded! Total molecules: {len(tox21_data)}")
        return tox21_data


    except StopIteration:
        print(f"❌ Error: No file matching 'tox21.csv*' found in {data_dir}")
        return None

if __name__ == '__main__':
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

