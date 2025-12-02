import pandas as pd
from sympy import plot
import matplotlib.pyplot as plt
    
df = pd.read_csv(r'D:/capture24/P001.csv', index_col='time', parse_dates=['time'],
                 dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotations': 'string'}  )
print(df)   

plt.figure(figsize=(10, 6))
plt.plot(df.index, df['x'], label='x')
plt.plot(df.index, df['y'], label='y')
plt.plot(df.index, df['z'], label='z')