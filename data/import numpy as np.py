import numpy as np
import pandas as pd

# Load the .npy file
data = np.load(r'C:\Users\ANKITH\OneDrive\Desktop\project_\ecg_web_app\data\test_demo.npy')

# Method 1: Using pandas (recommended for 2D arrays)
df = pd.DataFrame(data)
df.to_csv('output.csv', index=False)

# Method 2: Using numpy directly (for simple arrays)
# np.savetxt('output.csv', data, delimiter=',')