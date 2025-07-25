# Import Required Libraries
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

plt.ion()

# Load particles.json Data
with open('./data/DeCost-Holm_Data-in-Brief/d-lognormal-loc0.1-shape0.25/particles/particles1.json', 'r') as f:
    data = json.load(f)

particles = data['particles']
df = pd.DataFrame(particles)
distribution = data.get('distribution', 'unknown')
loc = data.get('loc', None)
shape = data.get('shape', None)
timestamp = data.get('timestamp', None)
df.head()

# 3D Scatter Plot of Particle Positions
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(df['x'], df['y'], df['z'], c=df['size'], s=df['size']*1e2, cmap='viridis')
plt.colorbar(sc, ax=ax, label='Particle Size')
ax.set_title('3D Scatter Plot of Particle Positions')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
input("Press Enter to continue to the next file...")
plt.close(fig)