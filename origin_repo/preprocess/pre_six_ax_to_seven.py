import pandas as pd

# Read file2.csv
file2 = pd.read_csv('../simon_trocar_feb_27/test_contact/joints/interpolated_all_joints.csv', header=None)

# Create file3 DataFrame as a copy of file2
file3_df = file2.copy()

# Delete the first column from file3
file3_df.drop(file3_df.columns[0], axis=1, inplace=True)

# Read file1.csv
file1_df = pd.read_csv('../simon_trocar_feb_27/test_contact/jaw/interpolated_all_jaw.csv', header=None)

# Insert file1 column 2 between file3 column 6 and 7
file3_df.insert(loc=6, column='New_Column_7', value=file1_df[file1_df.columns[1]])

# Insert file1 column 3 between file3 column 13 and 14
file3_df.insert(loc=13, column='New_Column_14', value=file1_df[file1_df.columns[2]])

# Copy file1 column 4 to the right of the last column of file3
file3_df.insert(loc=20, column='New_Column_21', value=file1_df[file1_df.columns[3]])

# Drop the first row from file3 DataFrame
file3_df = file3_df.rename(columns=file3_df.iloc[0]).drop(file3_df.index[0])

# Save file3 DataFrame to file3.csv
file3_df.to_csv('../simon_trocar_feb_27/test_contact/TO_cont_comp_test10_int.csv', index=False)
