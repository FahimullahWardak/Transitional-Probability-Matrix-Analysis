import pandas as pd
import numpy as np

# Example export data
data = {
    'Country': ['Afghanistan', 'India', 'USA', 'China'],
    '2008': [100, 200, 300, 400],
    '2009': [110, 210, 320, 410],
    '2010': [120, 220, 330, 420],
    '2011': [130, 230, 340, 430],
    '2012': [140, 240, 350, 440],
}

# Create DataFrame
df = pd.DataFrame(data)

# Transpose the DataFrame to have years as rows and countries as columns
df_transposed = df.set_index('Country').T

# Calculate the transition matrix
def calculate_transition_matrix(df):
    # Normalize data to get probabilities
    df_normalized = df.div(df.sum(axis=1), axis=0)
    transition_matrix = np.zeros((df.shape[1], df.shape[1]))

    for i in range(df.shape[0] - 1):
        current_year = df_normalized.iloc[i].values
        next_year = df_normalized.iloc[i + 1].values
        for j in range(len(current_year)):
            for k in range(len(next_year)):
                transition_matrix[j, k] += current_year[j] * next_year[k]
    
    # Normalize the transition matrix
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    
    return transition_matrix

transition_matrix = calculate_transition_matrix(df_transposed)

# Display the transition matrix
transition_matrix_df = pd.DataFrame(transition_matrix, index=df.columns[1:], columns=df.columns[1:])
print(transition_matrix_df)

# Predict the future state
def predict_future_state(current_state, transition_matrix, steps):
    state = np.array(current_state)
    for _ in range(steps):
        state = np.dot(state, transition_matrix)
    return state

# Example: Predict the state in 5 steps from the state in 2012
current_state = df_transposed.iloc[-1].values
future_state = predict_future_state(current_state, transition_matrix, 5)

# Display the future state
future_state_df = pd.Series(future_state, index=df.columns[1:])
print(future_state_df)
