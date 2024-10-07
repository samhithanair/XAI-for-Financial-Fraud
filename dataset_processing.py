import pandas as pd

df = pd.read_csv('Synthetic_Financial_datasets_log.csv')

# Filter rows where isFlaggedFraud == 0
flagged_fraud_0 = df[df['isFlaggedFraud'] == 0]

# Filter all rows where isFlaggedFraud == 1
flagged_fraud_1 = df[df['isFlaggedFraud'] == 1]

# Calculate how many rows we need to sample from flagged_fraud_0
required_rows = 500 - flagged_fraud_1.shape[0]

# Sample the required number of rows from flagged_fraud_0
if required_rows > 0:
    flagged_fraud_0_sample = flagged_fraud_0.sample(n=required_rows, replace=False)
else:
    flagged_fraud_0_sample = pd.DataFrame()  # No need to sample if we have enough flagged fraud cases

# Combine the two sets
combined_df = pd.concat([flagged_fraud_1, flagged_fraud_0_sample])

# Shuffle the final DataFrame
shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)

# Display or save the final DataFrame
shuffled_df.to_csv('train-data.csv', index=False)
