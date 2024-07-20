# DEL_TMP_VARS
import re

def del_tmp_vars(pattern="^tmp.*"):
    """
    Function to delete variables from the global namespace in a Jupyter notebook or Python environment,
    matching a specified regex pattern.

    Args:
    - pattern (str): Regular expression pattern to match variable names.
                     Default is "^tmp.*", which matches variables starting with 'tmp'.

    Example:
    # Suppose there are variables 'tmp_X', 'tmp_y', 'tmp_model' in the global namespace
    del_tmp_variables("^tmp.*") or del_tmp_variables()
    # Now 'tmp_X', 'tmp_y', 'tmp_model' are deleted from the global scope
    """
    # Get all variables in the global namespace
    global_vars = globals()
    # Find variables that match the specified pattern
    tmp_vars = [var for var in global_vars if re.match(pattern, var)]
    # Delete the matched variables
    for var in tmp_vars:
        del global_vars[var]

# CREATE_DATA_LOADERS
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def create_data_loaders(X, y, batch_size, train_percentage=0.2, shuffle=True, device='cpu', random_state=None):
    """
    Function to create DataLoaders for training and testing datasets.

    Args:
    - X (numpy array or pandas DataFrame): Input data.
    - y (numpy array or pandas Series): Output labels.
    - batch_size (int): Batch size for DataLoaders.
    - split_ratio (float): Ratio of the dataset to be used as testing data.
    - shuffle (bool): Whether to shuffle the data before splitting.
    - device (str): Device to place tensors ('cpu' or 'cuda').
    - random_state (int or None): Seed for reproducibility of data splitting.

    Returns:
    - train_loader (DataLoader): DataLoader for training dataset.
    - test_loader (DataLoader): DataLoader for testing dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_percentage, shuffle=shuffle, random_state=random_state)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# PLOT_CONFUSION_MATRIX
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_confusion_matrix(model, loader, num_separations=2):
    """
    Function to draw a confusion matrix for a PyTorch model.

    Args:
    - model (torch.nn.Module): PyTorch model to evaluate.
    - loader (torch.utils.data.DataLoader): DataLoader containing the dataset to evaluate.
    - rounding_choices (list of int, optional): List of choices to round predictions to (default: [0, 1]).

    Draws a confusion matrix by making predictions on the loader and comparing predicted vs true labels.
    """
    def round_to_nearest(value, choices):
        """
        Round a value to the nearest choice in a list.
        """
        return min(choices, key=lambda x: abs(x - value))

    rounding_choices = np.around(np.linspace(0, 1, num_separations), decimals=2).tolist()

    all_labels = []
    all_predictions = []

    # Initialize confusion matrix
    cm = np.zeros((2, num_separations), dtype=int)

    # Store expected and model outputs from the dataset
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            outputs = model(x_batch).squeeze(dim=-1)
            predictions = outputs.cpu().numpy()  # No sigmoid applied
            rounded_predictions = [round_to_nearest(pred, rounding_choices) for pred in predictions]
            all_labels.extend(y_batch.cpu().numpy())
            all_predictions.extend(rounded_predictions)

    # Ensure both all_labels and all_predictions are numpy arrays
    all_labels = np.array(all_labels, dtype=int)
    all_predictions = np.array(all_predictions, dtype=float)  # Use float for compatibility with rounding choices

    # Manually create the confusion matrix
    for true_label, pred_label in zip(all_labels, all_predictions):
        cm[true_label, rounding_choices.index(pred_label)] += 1

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=rounding_choices, yticklabels=[0, 1])
    plt.xlabel('Predicted')
    plt.ylabel('Expected')
    plt.title('Confusion Matrix')
    plt.show()

# TRAIN_MODEL

import torch
import torch.nn as nn
import torch.optim as optim
import time

def train_model(
    model, train_loader, test_loader, loss_function, optimizer,
    metric_function=None, epochs=1, debug_msg_interval=None,
    early_stop_epochs=None, min_improve_percentage = 0):
    """
    Train a PyTorch model using the specified train and test data loaders.

    Args:
    - model (torch.nn.Module): PyTorch model to train.
    - train_loader (torch.utils.data.DataLoader): DataLoader for training data.
    - test_loader (torch.utils.data.DataLoader): DataLoader for test/validation data.
    - loss_function: Loss function to optimize the model.
    - optimizer: Optimizer algorithm to update model parameters.
    - metric_function (function, optional): Function to calculate a metric for each batch during training and evaluation.
        Should accept outputs and y_batch as arguments.
    - epochs (int, optional): Number of training epochs (default: 10).
    - debug_msg_interval (int, optional): Interval for printing debug messages during training.
    - early_stop_epochs (int, optional): Number of epochs to wait for improvement in train loss before stopping early.
    - min_improve_percentage (float, optional): Minimum percentage of improvement in train loss to consider as an upgrade.
        Measured in percentage 0%-100%.

    Returns:
    - train_losses (list): List of training losses for each epoch.
    - test_losses (list): List of test/validation losses for each epoch.
    - metric_values (list): List of lists. Each inner list contains metric values calculated during evaluation for each epoch.
        metric_values[epoch][input_in_loader] gives the metric value for a specific input in a specific epoch.
    """
    start_time = time.time()
    train_losses = []
    test_losses = []
    metric_values = []
    min_loss = float("inf")
    epochs_with_no_upgrade=0
    for epoch in range(epochs):
        # Training part
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch).squeeze(dim=-1)
            loss = loss_function(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluation part
        model.eval()
        running_loss = 0.0
        epoch_metric_values = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                outputs = model(x_batch).squeeze(dim=-1)
                loss = loss_function(outputs, y_batch)
                running_loss += loss.item()

                if metric_function is not None:
                    metric_value = metric_function(outputs, y_batch)
                    epoch_metric_values.extend(metric_value)
        avg_test_loss = running_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        metric_values.append(epoch_metric_values)

        if debug_msg_interval is not None and (epoch + 1) % debug_msg_interval == 0:
            elapsed_time = time.time() - start_time
            print(f'({int(elapsed_time // 3600):02}:{int((elapsed_time % 3600) // 60):02}:{int(elapsed_time % 60):02})  '
                f'Epoch {epoch + 1}/{epochs}  '
                f'Train Loss: {avg_train_loss:.4f}  '
                f'Test Loss: {avg_test_loss:.4f}  '
                f'Min Loss: {min_loss:.4f}')     
        
        # Early stop
        if early_stop_epochs is not None:
            if  avg_train_loss < min_loss * (100.0 - min_improve_percentage) / 100.0:
                epochs_with_no_upgrade = 0
                min_loss = avg_train_loss
            else:
                epochs_with_no_upgrade += 1
                if epochs_with_no_upgrade >= early_stop_epochs:
                    print(f"Early stop at epoch {epoch + 1}, no improvement in {early_stop_epochs} epochs")
                    break
    return train_losses, test_losses, metric_values

# TEST MODEL
import torch

def test_model(model, data_loader, loss_function,  metric_function=None):
    """
    Test a PyTorch model on a given data loader.

    Args:
    - model (torch.nn.Module): PyTorch model to test.
    - data_loader (torch.utils.data.DataLoader): DataLoader for evaluation data.
    - metric_function (function): Function to calculate metrics.
    - loss_function: Loss function to evaluate the model.

    Returns:
    - test_loss (float): Average loss on the test set.
    - metric_value (float): Metric value calculated on the test set.
    """
    model.eval()
    test_loss = 0.0
    metric_values = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            outputs = model(x_batch).squeeze(dim=-1)
            loss = loss_function(outputs, y_batch)
            test_loss += loss.item()

            if metric_function is not None:
                metric_value = metric_function(outputs, y_batch)
                metric_values.extend(metric_value)

    test_loss /= len(data_loader)

    return test_loss, metric_values

# BALANCE_DATASET
import pandas as pd

def balance_dataset(df, target_column, shuffle=False, seed=42):
    """
    Balance a dataset by downsampling the majority class to match the number of samples in the minority class.

    Args:
    - df (pd.DataFrame): Pandas DataFrame containing the dataset.
    - target_column (pd.Series): Pandas Series representing the target variable (class labels).
    - shuffle (bool, optional): Whether to shuffle the balanced dataset (default: True).
    - seed (int, optional): Random seed for reproducibility (default: 42).

    Returns:
    - pd.DataFrame: Balanced DataFrame with an equal number of samples for each class.
    """
    # Count the number of samples in each class
    class_counts = target_column.value_counts()
    minority_class = class_counts.idxmin()
    minority_count = class_counts.min()

    # Downsample the majority class
    df_majority_downsampled = df[target_column != minority_class].sample(n=minority_count, random_state=seed)

    # Keep all samples from the minority class
    df_minority = df[target_column == minority_class]

    # Combine the downsampled and minority DataFrames
    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    # Shuffle the balanced dataset if shuffle=True
    if shuffle:
        df_balanced = df_balanced.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df_balanced

# PLOT_PREDICTIONS_DENSITY

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_predictions_density(model, test_loader):
    """
    Plot the density of model predictions on the test set.

    Args:
    - model (torch.nn.Module): PyTorch model to evaluate.
    - test_loader (torch.utils.data.DataLoader): DataLoader for the test/validation data.

    The function evaluates the model on the test set, collects the predictions,
    and plots the density of these predictions. If the variance of the predictions
    is zero, it suggests using a different plot type.
    """
    all_labels = []
    all_predictions = []

    # Evaluate the model and get predictions from the test set
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs = model(x_batch).squeeze(dim=-1)
            predictions = outputs
            all_labels.extend(y_batch.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Create a DataFrame for visualization with seaborn
    df = pd.DataFrame({
        'Labels': all_labels,
        'Predictions': all_predictions
    })

    # Calculate the variance of the predictions
    predictions_variance = np.var(df['Predictions'])

    # Plot the density plot if the variance is greater than zero
    plt.figure(figsize=(10, 6))
    if predictions_variance > 0:
        sns.kdeplot(df['Predictions'], fill=True)
        plt.axvline(0.5, color='red', linestyle='--')  # Add vertical line at 0.5
        plt.xlabel('Predicted Values')
        plt.ylabel('Density')
        plt.title('Density Plot of Predicted Values')
        plt.show()
    else:
        print("Predictions have zero variance. Consider using a different plot type.")

# SAVE_DATASET

import os
import shutil
import pandas as pd

def save_dataset(df, filename, file_len):
    """
    Save a DataFrame to multiple CSV files, each containing a specified number of rows. 
    The files are saved in a directory named after the filename, which is created or 
    cleaned if it already exists. Finally, the directory is compressed into a zip file.

    Args:
    - df (pd.DataFrame): The DataFrame to be saved.
    - filename (str): The base filename for the saved CSV files.
    - file_len (int): The number of rows per CSV file.

    Example:
    save_dataset(df, 'dataset', 1000)
    # This will save the DataFrame 'df' into multiple CSV files, each containing 1000 rows, 
    # in a directory named 'dataset', and then compress the directory into 'dataset.zip'.
    """
    # Remove '.csv' extension from filename if it exists
    if filename.endswith('.csv'):
        filename = filename[:-4]

    # Create folder if it doesn't exist, or clean it if it does
    if os.path.exists(filename):
        # Remove all contents of the directory
        shutil.rmtree(filename)
    os.makedirs(filename)

    # Save the DataFrame in chunks of file_len rows
    num_slices = df.shape[0] // file_len + (1 if df.shape[0] % file_len != 0 else 0)
    for i in range(num_slices):
        df_slice = df.iloc[i*file_len:(i+1)*file_len]
        df_slice.to_csv(f'{filename}/{filename}_{i}.csv', index=False)

    # Compress the directory into a zip file
    shutil.make_archive(filename, 'zip', filename)

# LOAD_DATASET

import os
import pandas as pd

def load_dataset(dataset_name):
    """
    Load a dataset from a directory containing multiple CSV files. If the directory
    does not exist, it attempts to unzip a file with the same base name.

    Args:
    - dataset_name (str): The base name of the dataset directory (and zip file if needed).

    Returns:
    - df (pd.DataFrame): The concatenated DataFrame containing the data from all CSV files.

    Example:
    df = load_dataset('dataset')
    # This will load the DataFrame 'df' from the directory 'dataset', unzipping 'dataset.zip' if necessary.
    """
    # Check if the dataset directory exists, if not, unzip the corresponding zip file
    if not os.path.exists(dataset_name):
        print(f'Dataset {dataset_name} not found. Downloading...')
        os.system(f'unzip -q {dataset_name}.zip')
        print(f'Dataset {dataset_name} downloaded.')

    # Get and sort the filenames in the dataset directory
    files = os.listdir(dataset_name)
    files.sort()
    files = [os.path.join(dataset_name, file) for file in files]

    # Concatenate all CSV files into a single DataFrame
    df = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    
    return df

# COUNT_COLUMNS

import csv

def count_columns(csv_file):
    """
    Count the number of columns in a given CSV file without reading the entire file.

    Args:
    - csv_file (str): Path to the CSV file.

    Returns:
    - num_columns (int): Number of columns in the CSV file.

    Example:
    num_columns = count_columns('data.csv')
    # This will return the number of columns in 'data.csv'.
    """
    # Open the CSV file and read the first line to count the columns
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        first_row = next(reader)
        num_columns = len(first_row)
    
    return num_columns


# SAVE_MODEL
import os
import re
import torch
import shutil

def save_model(model, name, notebook_path):
    """
    Saves a PyTorch model with an incremented index in a specified directory.
    
    Args:
        model (torch.nn.Module): The PyTorch model to save.
        name (str): The base name or path for saving the model.
        
    Returns:
        str: The full path where the model is saved.
        str: The full path where the notebook is saved.
    
    Example:
        model = MyModel()
        model_path = save_model(model, "models/model")
        print(model_path)  # Output might be "models/model_0.pth" or "models/model_1.pth" if "models/model_0.pth" already exists.
    """
    
    # 0- Remove the extension if it exists
    name = os.path.splitext(name)[0]

    # 1- Create the directory if it does not exist
    if not os.path.exists(name):
        print(f"Creating directory {name}")
        os.makedirs(name)

    # 2- Find the highest index of existing models in the directory
    max_index = -1
    pattern = re.compile(f'{re.escape(name)}_([0-9]+)\\.pth')
    
    for filename in os.listdir(name):
        match = pattern.match(filename)
        if match:
            index = int(match.group(1))
            if index > max_index:
                max_index = index

    # 3- Create the new model name with incremented index
    new_index = max_index + 1
    new_name = os.path.join(name, f"{os.path.basename(name)}_{new_index}.pth")

    # 4- Save the model using torch.save
    torch.save(model.state_dict(), new_name)

    # 5- Save the ipynb too
    nb_name = os.path.join(name, f"{os.path.basename(name)}_{new_index}.ipynb")
    shutil.copy(notebook_path, nb_name)

    return new_name, nb_name

# PROMPT_USER

def prompt_user(prompt, allowed_responses = ["yes", "no", "y", "n"]):
    """
    Prompts the user with a message until a valid response from allowed_responses is given.
    
    Args:
        prompt (str): The message to display to the user.
        allowed_responses (list of str): A list of valid responses.
        
    Returns:
        str: The valid response from the user.
    """
    response = ""
    while response not in allowed_responses:
        response = input(prompt).strip().lower()
    return response