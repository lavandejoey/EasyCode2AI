import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting

# Step 1: Generate 3D sample data
np.random.seed(0)
mean = [0, 0, 0]
cov = [[5, 2, 1],
       [2, 3, 0.5],
       [1, 0.5, 2]]

data_3d = np.random.multivariate_normal(mean, cov, 50)

# Step 2: Fit PCA for 2 components (3D -> 2D)
pca = PCA(n_components=2)
pca.fit(data_3d)
data_2d = pca.transform(data_3d)

# Step 3: Get principal components and mean
pc1 = pca.components_[0]
pc2 = pca.components_[1]
data_mean = pca.mean_

# Step 4: Plotting
fig = plt.figure(figsize=(14, 6))

# Left plot: Original 3D data + PC1 and PC2 vectors
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], color='blue', alpha=0.6)

# Draw PC1 line
scale = 10  # Just for visualization scaling
ax1.quiver(*data_mean, *pc1, length=scale, color='red', label='PC1', linewidth=2, arrow_length_ratio=0.1)
# Draw PC2 line
ax1.quiver(*data_mean, *pc2, length=scale, color='green', label='PC2', linewidth=2, arrow_length_ratio=0.1)

ax1.set_title('Original 3D Data + Principal Components')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Equal aspect ratio
max_range = np.array([data_3d[:, 0].max() - data_3d[:, 0].min(),
                      data_3d[:, 1].max() - data_3d[:, 1].min(),
                      data_3d[:, 2].max() - data_3d[:, 2].min()]).max() / 2.0

mid_x = (data_3d[:, 0].max() + data_3d[:, 0].min()) * 0.5
mid_y = (data_3d[:, 1].max() + data_3d[:, 1].min()) * 0.5
mid_z = (data_3d[:, 2].max() + data_3d[:, 2].min()) * 0.5

ax1.set_xlim(mid_x - max_range, mid_x + max_range)
ax1.set_ylim(mid_y - max_range, mid_y + max_range)
ax1.set_zlim(mid_z - max_range, mid_z + max_range)

ax1.legend()

# Right plot: Projected 2D data
ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(data_2d[:, 0], data_2d[:, 1], color='purple', alpha=0.6)

ax2.set_title('Projected 2D Data (PCA)')
ax2.set_xlabel('1st Principal Component')
ax2.set_ylabel('2nd Principal Component')
ax2.axis('equal')
ax2.grid(True)

plt.suptitle('PCA: 3D Data with PC1 & PC2 Vectors and 2D Projection', fontsize=16)
plt.tight_layout()
plt.show()
