import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Step 1: Generate 2D sample data
np.random.seed(0)
mean = [2, 3]
cov = [[3, 1], [1, 2]]  # Covariance matrix
data = np.random.multivariate_normal(mean, cov, 20)

# Step 2: Fit PCA for 1 component (2D -> 1D)
pca = PCA(n_components=1)
pca.fit(data)

# Get principal axis (first principal component direction)
principal_axis = pca.components_[0]
data_mean = pca.mean_

# Step 3: Project data onto the principal component line
projected_data_1d = pca.transform(data)  # 1D coordinates
projected_data_2d = pca.inverse_transform(projected_data_1d)  # Back to 2D for plotting

# Step 4: Define axis limits to keep both charts consistent
x_min = min(data[:, 0].min(), projected_data_2d[:, 0].min()) - 1
x_max = max(data[:, 0].max(), projected_data_2d[:, 0].max()) + 1
y_min = min(data[:, 1].min(), projected_data_2d[:, 1].min()) - 1
y_max = max(data[:, 1].max(), projected_data_2d[:, 1].max()) + 1

# Step 5: Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Left Chart: Original points only
axs[0].scatter(data[:, 0], data[:, 1], alpha=0.5, label='Original Data', color='blue')
axs[0].set_title('Original Data')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].set_xlim(x_min, x_max)
axs[0].set_ylim(y_min, y_max)
axs[0].axis('equal')
axs[0].grid(True)
axs[0].legend()

# Right Chart: PCA with projections
axs[1].scatter(data[:, 0], data[:, 1], alpha=0.5, label='Original Data', color='blue')

# Plot the principal component line
line_length = 8
line_points = np.array([data_mean - line_length * principal_axis,
                        data_mean + line_length * principal_axis])
axs[1].plot(line_points[:, 0], line_points[:, 1], color='red', label='Principal Component')

# Plot projections on the PCA line
axs[1].scatter(projected_data_2d[:, 0], projected_data_2d[:, 1], color='green', label='Projected Points')

# Lines connecting original points to their projections
for original, projected in zip(data, projected_data_2d):
    axs[1].plot([original[0], projected[0]], [original[1], projected[1]], 'gray', linewidth=0.5)

axs[1].set_title('PCA Projection (2D -> 1D)')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')
axs[1].set_xlim(x_min, x_max)
axs[1].set_ylim(y_min, y_max)
axs[1].axis('equal')
axs[1].grid(True)
axs[1].legend()

plt.suptitle('PCA Visualization: 2D Data and 1D Projection', fontsize=16)
plt.tight_layout()
plt.show()
