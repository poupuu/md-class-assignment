#EDA and Preprocessing
eda_preprocessor = EDAandPreprocessing(input_df, output_df)

#check data types
print("\n--- Checking Data Types ---")
for column in input_df.columns:
    dtype = eda_preprocessor.checkDataType(column)
    print(f"Column: {column}, Data Type: {dtype}")

#convert columns to numeric (if needed)
print("\n--- Converting Columns to Numeric ---")
numeric_columns = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"] #configurable
for column in numeric_columns:
    if column in input_df.columns:
        eda_preprocessor.dataConvertToNumeric(column)
        print(f"Converted '{column}' to numeric type.")

#convert columns to categorical (if needed)
print("\n--- Converting Columns to Categorical ---")
categorical_columns = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"] #configurable
for column in categorical_columns:
    if column in input_df.columns:
        eda_preprocessor.dataConvertToCategorical(column)
        print(f"Converted '{column}' to categorical type.")

#check for missing values
print("\n--- Checking for Missing Values ---")
missing_values = {}
for column in input_df.columns:
    missing_count = eda_preprocessor.checkMissingValue(column)
    missing_values[column] = missing_count
    print(f"Column: {column}, Missing Values: {missing_count}")

#handle missing values
print("\n--- Handling Missing Values ---")
for column, count in missing_values.items():
    if count > 0:
        eda_preprocessor.handleMissingValue(column)
        print(f"Handled missing values in '{column}'.")

#check distribution of columns
print("\n--- Checking Distribution of Columns ---")
for column in input_df.columns:
    print(f"\nDistribution of Column: {column}")
    eda_preprocessor.checkDistribution(column)

#ordinal encoding for target variable (manual input)
print("\n--- Ordinal Encoding for Target Variable ---")
target_categories = ["Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", 
                     "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"] #configurable

#manual ordinal mapping
ordinal_mapping = {
    "Insufficient_Weight": 0,
    "Normal_Weight": 1,
    "Overweight_Level_I": 2,
    "Overweight_Level_II": 3,
    "Obesity_Type_I": 4,
    "Obesity_Type_II": 5,
    "Obesity_Type_III": 6
} #configurable

#apply ordinal encoding
output_df = output_df.map(ordinal_mapping)

#save encoded target variable
FileHandler.saveEncoder(output_df, "encoded_target_variable.pkl")
print("Saved encoded target variable.")

#print after encode target variable
print("Ordinal Encoded Target Variable:")
print(output_df.value_counts().sort_index())

#label encoding for categorical features
print("\n--- Label Encoding for Categorical Features ---")

#identify categorical columns in the input data
categorical_columns = input_df.select_dtypes(include=['object', 'category']).columns.tolist()
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    input_df[column] = le.fit_transform(input_df[column])
    label_encoders[column] = le  
    print(f"Applied Label Encoding to Column: {column}")

#save label encoder features
FileHandler.saveEncoder(label_encoders, "label_encoders.pkl")
print("Saved label encoders.")

#print after encode features
print("\nLabel Encoded Input Data:")
print(input_df.head())

#splitting features and target (not really needed, just us input_df and output_df)
# print("\n--- Creating Input and Output Data ---")
# input_df, output_df = data_handler.create_input_output("NObeyesdad")

# # Print shapes of input_df and output_df
# print(f"Shape of Features (input_df): {input_df.shape}")
# print(f"Shape of Target (output_df): {output_df.shape}")

# # Display the first few rows of input_df and output_df
# print("\nFirst Few Rows of Features (input_df):")
# print(input_df.head())

# print("\nFirst Few Rows of Target (output_df):")
# print(output_df.head())

#feature scaling (standard scaling and robust scaling)
print("\n--- Feature Scaling ---")
standard_scaling_columns = ["Height"]  #configurable
robust_scaling_columns = ["Age", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]  #configurable

#standard
if standard_scaling_columns:
    print("Applying Standard Scaling to columns:", standard_scaling_columns)
    eda_preprocessor.standardScaling(standard_scaling_columns)

#robust
if robust_scaling_columns:
    print("Applying Robust Scaling to columns:", robust_scaling_columns)
    eda_preprocessor.robustScaling(robust_scaling_columns)

standard_scaler = StandardScaler()
robust_scaler = RobustScaler()

#fit standard scaler
if standard_scaling_columns:
    print("Applying Standard Scaling to columns:", standard_scaling_columns)
    eda_preprocessor.fitScaler(standard_scaler, standard_scaling_columns)

#fit robust scaler
if robust_scaling_columns:
    print("Applying Robust Scaling to columns:", robust_scaling_columns)
    eda_preprocessor.fitScaler(robust_scaler, robust_scaling_columns)

#save scalers
FileHandler.saveScaler(standard_scaler, "standard_scaler.pkl")
FileHandler.saveScaler(robust_scaler, "robust_scaler.pkl")
print("Saved standard and robust scalers.")

#final check
print("\nFirst Few Rows of Scaled Features (X):")
print(input_df.head())
print(output_df.head())
