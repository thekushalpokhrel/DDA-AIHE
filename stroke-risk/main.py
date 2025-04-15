import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix  # Import confusion_matrix
from sklearn.preprocessing import StandardScaler # Import StandardScaler
from imblearn.over_sampling import SMOTE # Import SMOTE

# Task 1: Import CSV into pandas
def process_data(file_path):
    """
    Reads a CSV file into a pandas DataFrame and removes rows with missing values.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing the cleaned DataFrame and a dictionary of column data types.
    """
    df = pd.read_csv(file_path)
    df = df.dropna()
    df_types = {col: dtype for col, dtype in zip(df.columns, df.dtypes)}
    return df, df_types

# Task 2: Convert Categorical Variables to Numeric
def convert_to_numeric(file_path="stroke-data.csv"):
    """
    Converts categorical variables in a DataFrame to numeric using one-hot encoding.

    Args:
        file_path (str, optional): The path to the CSV file. Defaults to "stroke-data.csv".

    Returns:
        pandas.DataFrame: The DataFrame with categorical variables converted to numeric.
    """
    df, dict_types = process_data(file_path)
    cat_ls = [col for col, dtype in dict_types.items() if dtype == 'object']
    for col in cat_ls:
        unique_vals = df[col].unique()
        if len(unique_vals) == 2:
            df[col] = pd.get_dummies(df[col], drop_first=True, dtype=int)
        else:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            df = pd.concat([df, dummies], axis=1).drop(columns=[col])
    if 'stroke' in df.columns:
        stroke_col = df.pop('stroke')
        df['stroke'] = stroke_col
    return df

# Task 3: Generate ndarrays for Train and Test Data
def create_arrays(file_path="stroke-data.csv", test_size=0.33, random_state=42):
    """
    Splits the data into training and testing sets.  Scales numerical features and applies SMOTE.

    Args:
        file_path (str, optional): Path to the CSV file. Defaults to "stroke-data.csv".
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.33.
        random_state (int, optional): The seed for the random number generator. Defaults to 42.

    Returns:
        tuple: A tuple containing the training and testing sets for X and y.
    """
    df = convert_to_numeric(file_path)
    X = df.drop(columns=['id', 'stroke'], errors='ignore')
    y = df['stroke'].values

    # Identify numerical columns for scaling
    numerical_cols = X.select_dtypes(include=np.number).columns
    X_numerical = X[numerical_cols]

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numerical)
    X_scaled_df = pd.DataFrame(X_scaled, index=X_numerical.index, columns=numerical_cols)

    # Drop original numerical columns and concatenate scaled ones
    X = X.drop(columns=numerical_cols)
    X = pd.concat([X, X_scaled_df], axis=1)

    X = X.values  # Convert to numpy array AFTER scaling and handling categorical

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Apply SMOTE to the training data only
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return X_train_resampled, X_test, y_train_resampled, y_test


# Task 4: Create the Logistic Regression Model
def fit_logitmodel(X, y, penalty='l2', solver='liblinear', max_iter=100):
    """
    Fits a logistic regression model to the training data.

    Args:
        X (numpy.ndarray): The training data features.
        y (numpy.ndarray): The training data labels.
        penalty (str, optional): The norm used in the penalization. Defaults to 'l2'.
        solver (str, optional): The algorithm to use in the optimization problem. Defaults to 'liblinear'.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        sklearn.linear_model.LogisticRegression: The trained logistic regression model.
    """
    logitmodel_stroke = LogisticRegression(penalty=penalty, solver=solver, max_iter=max_iter)
    logitmodel_stroke.fit(X, y)
    return logitmodel_stroke

# Task 5: Model Evaluation
def evaluate_logitmodel(model, X_test, y_true):
    """
    Evaluates the performance of a logistic regression model.

    Args:
        model (sklearn.linear_model.LogisticRegression): The trained logistic regression model.
        X_test (numpy.ndarray): The testing data features.
        y_true (numpy.ndarray): The true labels for the testing data.

    Returns:
        tuple: A tuple containing the accuracy, precision, and recall of the model on the test data.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)  # Handle potential division by zero
    recall = recall_score(y_true, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)
    return acc, prec, recall, conf_matrix

if __name__ == "__main__":
    # Set the file path
    file_path = "stroke-data.csv"
    # Create the arrays
    X_tr, X_ts, y_tr, y_ts = create_arrays(file_path)
    print(f"X_train shape: {X_tr.shape}, X_test shape: {X_ts.shape}, y_train shape: {y_tr.shape}, y_test shape: {y_ts.shape}")

    # Fit the logistic regression model
    logit_m = fit_logitmodel(X_tr, y_tr)

    # Evaluate the model
    acc, prec, recall, conf_matrix = evaluate_logitmodel(logit_m, X_ts, y_ts)
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {recall:.4f}")
    print("Confusion Matrix:\n", conf_matrix)