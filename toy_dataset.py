import pandas as pd
import itertools
import numpy as np

def get_toy_dataset():
    x_vals = range(1, 5)
    y_vals = range(1, 5)
    combinations = list(itertools.product(x_vals, y_vals))

    # Create DataFrame
    df = pd.DataFrame(combinations, columns=['x', 'y'])

    # Add random values for 'Cu' (for example, random floats between 0 and 100)
    df['Cu'] = np.random.uniform(0, 100, size=len(df))
    return df