# Project: Predicting Molecular Properties with Linear Regression

# --- Imports ---
import matplotlib
import warnings
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np # For np.allclose check
import argparse # For command-line arguments

# Configure Matplotlib backend at top for non-interactive plotting
matplotlib.use('Agg')

# --- Function Definitions ---

def calculate_rdkit_descriptors(smiles_list_cleaned, smiles_column_name, df_molecules_clean, custom_descriptor_list=None):
    """
    Converts SMILES to RDKit Mol objects, filters invalid ones,
    and calculates a comprehensive set of RDKit descriptors.
    Can use a custom list of descriptors.
    """
    mol_objects = [Chem.MolFromSmiles(s) for s in smiles_list_cleaned]
    valid_mols = [m for m in mol_objects if m is not None]

    # Align the DataFrame with the valid_mols
    original_indices_of_valid_mols = [i for i, m in enumerate(mol_objects) if m is not None]
    df_molecules_final_aligned = df_molecules_clean.iloc[original_indices_of_valid_mols].reset_index(drop=True)

    print(f"\nSuccessfully converted {len(valid_mols)} molecules out of {len(smiles_list_cleaned)} cleaned SMILES strings.")
    print(f"DataFrame aligned to valid molecules (df_molecules_final_aligned) has {len(df_molecules_final_aligned)} rows.")

    # Use the custom_descriptor_list if provided, otherwise use the default Descriptors.descList
    descriptors_to_calculate = custom_descriptor_list if custom_descriptor_list is not None else Descriptors.descList

    descriptor_data = []

    for i, mol in enumerate(valid_mols):
        row = {}
        for name, func in descriptors_to_calculate:
            try:
                row[name] = func(mol)
            except Exception:
                row[name] = None
        descriptor_data.append(row)

    df_descriptors = pd.DataFrame(descriptor_data)
    df_descriptors['SMILES'] = df_molecules_final_aligned[smiles_column_name] # Add SMILES back for reference

    print(f"\nCalculated descriptors for {len(df_descriptors)} molecules.")
    return df_descriptors

def preprocess_data(df_descriptors, target_property, features):
    """
    Applies outlier removal (IQR), handles NaNs, and scales features.
    Returns cleaned X and y DataFrames.
    """
    # --- Outlier Removal (using IQR method on the target property) ---
    if target_property not in df_descriptors.columns:
        raise KeyError(f"Target property '{target_property}' not found in descriptor DataFrame.")

    Q1 = df_descriptors[target_property].quantile(0.25)
    Q3 = df_descriptors[target_property].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_filtered_iqr = df_descriptors[
        (df_descriptors[target_property] >= lower_bound) &
        (df_descriptors[target_property] <= upper_bound)
    ]

    print(f"\nRows in df_descriptors before IQR filtering: {len(df_descriptors)}")
    print(f"Removed {len(df_descriptors) - len(df_filtered_iqr)} rows based on '{target_property}' IQR outliers.")
    print(f"Rows after IQR filtering: {len(df_filtered_iqr)}")

    # --- Final Cleaning for ML (handling NaNs in features or target) ---
    df_clean_for_ml = df_filtered_iqr.dropna(subset=[target_property] + features)

    X = df_clean_for_ml[features]
    y = df_clean_for_ml[target_property]

    print(f"\nShape of X (features) before variance check: {X.shape}")
    print(f"Number of features selected: {len(X.columns)}")

    # Check variance of each feature and remove zero-variance ones
    feature_variances = X.var()
    zero_variance_features = feature_variances[feature_variances == 0].index.tolist()

    if zero_variance_features:
        print(f"\nWARNING: The following {len(zero_variance_features)} features have zero variance and will be removed:")
        print(zero_variance_features)
        X = X.drop(columns=zero_variance_features)
        print(f"New shape of X after removing zero-variance features: {X.shape}")
    else:
        print("\nAll selected features have non-zero variance.")

    print(f"\nVariance of target (y): {y.var():.4f}")
    if y.var() == 0:
        print("WARNING: Target variable (y) has zero variance. This means all MolLogP values are the same.")

    # --- Apply Feature Scaling ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    print(f"\nFeatures scaled using StandardScaler. New shape: {X.shape}")
    print(f"Final DataFrame for ML (X, y) prepared with {len(X)} rows after dropping any remaining NaNs.")
    return X, y

def main():
    # --- Argparse Setup ---
    parser = argparse.ArgumentParser(description="Calculate molecular properties and predict a target property using Linear Regression.")
    parser.add_argument('csv_filepath', type=str,
                        help="Path to the CSV file containing SMILES strings (e.g., 'data/molecules.csv').")
    parser.add_argument('--smiles_col', type=str, default='Smiles',
                        help="Name of the column containing SMILES strings in the CSV. Default is 'Smiles'.")
    parser.add_argument('--sep', type=str, default=';',
                        help="Delimiter used in the CSV file. Default is ';'.")
    parser.add_argument('--target_prop', type=str, default='MolLogP',
                        choices=['MolLogP', 'MolWt', 'TPSA', 'NumHDonors', 'MolMR'],
                        help="The RDKit-calculated molecular property to predict. Recommended: MolLogP (Lipophilicity), MolWt (Molecular Weight), TPSA (Polar Surface Area), NumHDonors (H-Bond Donors), MolMR (Molar Refractivity). Default is 'MolLogP'.")
    parser.add_argument('--output_plot_name', type=str,
                        help="Optional: Name for the output plot file (e.g., 'my_plot.png'). If not provided, plot will be saved as '{target_property}_prediction_plot.png'.")

    args = parser.parse_args()

    csv_filepath = args.csv_filepath
    smiles_column_name = args.smiles_col
    csv_delimiter = args.sep
    target_property = args.target_prop
    output_plot_name = args.output_plot_name if args.output_plot_name else f'{target_property}_prediction_plot.png'

    # --- Prepare the list of RDKit Descriptors, excluding known problematic ones ---
    # This filter aims to remove descriptors that often cause the MorganGenerator warning
    # by internally calling deprecated RDKit fingerprinting methods.
    # The common culprits are often BCUT2D_* and some EStateIndex variants.
    # FpDensityMorgan1, 2, and 3 are also explicitly excluded now.
    filtered_descriptor_list = []
    for name, func in Descriptors.descList:
        # Exclude FpDensityMorgan descriptors
        if not name.startswith('FpDensityMorgan'):
            filtered_descriptor_list.append((name, func))
    print(f"\nFiltered descriptor list to {len(filtered_descriptor_list)} descriptors to avoid MorganGenerator warnings.")

    # --- Main Script Logic ---

    # --- Step 1: Importing and cleaning the CSV file ---
    try:
        df_molecules = pd.read_csv(csv_filepath, sep=csv_delimiter, on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading CSV file '{csv_filepath}': {e}")
        sys.exit(1)

    print("File loading successful!")
    print("Columns in the DataFrame:", df_molecules.columns.tolist())

    if smiles_column_name not in df_molecules.columns:
        print(f"Error: SMILES column '{smiles_column_name}' not found in the CSV. Available columns: {df_molecules.columns.tolist()}")
        sys.exit(1)

    smiles_list_raw = df_molecules[smiles_column_name].tolist()
    print(f"Total SMILES in list (raw): {len(smiles_list_raw)}")

    non_string_count = 0
    empty_string_count = 0
    nan_count = 0
    for s in smiles_list_raw:
        if not isinstance(s, str):
            non_string_count += 1
            if pd.isna(s):
                nan_count += 1
        elif s == '':
            empty_string_count += 1
    print(f"\nSummary of raw SMILES list issues:")
    print(f"Non-string values (total): {non_string_count}")
    print(f"NaN values (subset of non-string): {nan_count}")
    print(f"Empty strings: {empty_string_count}")

    initial_rows = len(df_molecules)
    df_molecules_clean = df_molecules.dropna(subset=[smiles_column_name])
    df_molecules_clean = df_molecules_clean[df_molecules_clean[smiles_column_name] != '']
    cleaned_rows = len(df_molecules_clean)

    print(f"\nInitial rows in DataFrame: {initial_rows}")
    print(f"Removed {initial_rows - cleaned_rows} rows due to missing or empty SMILES.")
    print(f"Rows after initial SMILES cleaning: {cleaned_rows}")

    smiles_list_cleaned = df_molecules_clean[smiles_column_name].tolist()
    print(f"SMILES list size after cleaning: {len(smiles_list_cleaned)}")

    # --- Calculate RDKit Descriptors ---
    try:
        # Pass the filtered list to the descriptor calculation function
        df_descriptors = calculate_rdkit_descriptors(smiles_list_cleaned, smiles_column_name, df_molecules_clean, custom_descriptor_list=filtered_descriptor_list)
    except Exception as e:
        print(f"Error during RDKit descriptor calculation: {e}")
        sys.exit(1)

    # --- Step 2: Prepare Data for Scikit-learn ---
    initial_features = [col for col in df_descriptors.columns if col != target_property and pd.api.types.is_numeric_dtype(df_descriptors[col])]

    X, y = preprocess_data(df_descriptors.copy(), target_property, initial_features)

    print(f"\nFinal DataFrame for ML (X, y) prepared with {len(X)} rows.")
    print(f"Number of features (descriptors) used for training: {len(X.columns)}")


    # --- Step 3: Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    print(f"\n--- Verifying Data Before Model Training ---")
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")

    train_feature_variances = X_train.var()
    zero_variance_train_features = train_feature_variances[train_feature_variances == 0].index.tolist()
    if zero_variance_train_features:
        print(f"\nCRITICAL WARNING: The following features have zero variance in X_train:")
        print(zero_variance_train_features)
        print("This means the model has no information from these features in training.")
    else:
        print("\nAll features in X_train have non-zero variance. Good.")

    print(f"\nVariance of y_train: {y_train.var():.4f}")
    if y_train.var() == 0:
        print("CRITICAL WARNING: y_train has zero variance. All target values are the same in training set.")
    else:
        print("y_train has non-zero variance. Good.")

    print(f"\nTraining set size: {len(X_train)} molecules")
    print(f"Test set size: {len(X_test)} molecules")


    # --- Step 4: Train a Scikit-learn model ---
    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"\nScikit-learn Linear Regression model trained.")

    # --- Analyze Feature Importance (Coefficients) ---
    print("\n--- Feature Importance (Linear Regression Coefficients) ---")

    feature_names_trained = X_train.columns.tolist()
    coefficients = model.coef_

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names_trained,
        'Coefficient': coefficients
    })

    feature_importance_df['Abs_Coefficient'] = abs(feature_importance_df['Coefficient'])
    feature_importance_df = feature_importance_df.sort_values(by='Abs_Coefficient', ascending=False)

    top_n = 10
    print(f"Top {top_n} Most Influential Descriptors for {target_property}:")
    print(feature_importance_df.head(top_n))
    print(f"\nModel Intercept: {model.intercept_:.4f}")


    # --- Step 5: Evaluate the model ---
    y_pred = model.predict(X_test)

    print(f"\n--- Verifying Predictions ---")
    print(f"Shape of y_pred: {y_pred.shape}")
    print(f"First 10 values of y_pred: {y_pred[:10]}")
    print(f"Min predicted value: {y_pred.min():.4f}")
    print(f"Max predicted value: {y_pred.max():.4f}")
    print(f"Variance of y_pred: {y_pred.var():.4f}")
    if y_pred.var() == 0:
        print("CRITICAL WARNING: y_pred has zero variance. All predictions are the same!")

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Scikit-learn Model Performance ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R2) score: {r2:.2f}")

    # --- Showcase: Visualization and Plot Saving ---
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, s=20)

    plot_min = min(y_test.min(), y_pred.min())
    plot_max = max(y_test.max(), y_pred.max())
    plt.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', lw=2)

    plt.xlabel(f'Actual {target_property}')
    plt.ylabel(f'Predicted {target_property}')
    plt.title(f'Actual vs. Predicted {target_property} (Linear Regression)')
    plt.grid(True)

    # Save the plot
    plt.savefig(output_plot_name, dpi=300)
    plt.close()

    print(f"\nPlot saved as '{output_plot_name}'")

# --- Main execution block ---
if __name__ == "__main__":
    main()