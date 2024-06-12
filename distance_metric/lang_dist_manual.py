import pandas as pd

#trying with the toy manual feature set (6 fts)
df = pd.read_csv('features_manual.csv')


# Set the ID column as the index
df.set_index('Language', inplace=True)
reference_row = df.loc['eng']
results = {}


#im setting it up as % diff entries this time because there are NaNs etc
for index, row in df.iterrows():
    if index == '001':
        continue
    
    total_entries = 0
    different_entries = 0
    
    # Compare each entry in the row to the reference entry
    for column in df.columns:
        value = row[column]
        if value != '':
            total_entries += 1
            if value != reference_row[column]:
                different_entries += 1
    
    # Calculate the percentage of different entries
    if total_entries > 0:
        difference_percentage = (different_entries / total_entries) * 100
    else:
        difference_percentage = 0
    
    # Store the result
    results[index] = difference_percentage


# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(list(results.items()), columns=['Language', '% distance'])
results_df = results_df.sort_values(by=results_df.columns[1])

print(results_df)
