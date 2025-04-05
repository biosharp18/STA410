import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
cmap = ListedColormap(['black', 'green', 'blue', "yellow", "red"])

def edit_heatmap(edits):
  #Edits shape (n_iterations, X.shape[2])
  edits_copy = edits.copy() + 1
  plt.imshow(edits_copy[:, :], aspect='auto', cmap=cmap)
  plt.xlabel("Position")
  plt.ylabel("Iteration")
  plt.title("Edit Heatmap")

  legend_handles = [
    mpatches.Patch(color='green', label='A'),
    mpatches.Patch(color='blue', label='C'),
    mpatches.Patch(color='yellow', label='G'),
    mpatches.Patch(color='red', label='T')
  ]
  # Add color bar

# Add legend to the plot
  plt.legend(handles=legend_handles, title='Nucleotide', loc='upper right')
  plt.show()



def plot_unit_vectors():
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    labels = ["A", "C", "G"]
    colors = ["r", "g", "b"]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the vectors
    origin = np.zeros((3,))  # Origin point
    for vec, label, color in zip(vectors, labels, colors):
        ax.quiver(*origin, *vec, color=color, arrow_length_ratio=0.2)
        ax.text(*vec, label, fontsize=12, color=color)

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set axis limits
    ax.set_xlim([0, 1.2])
    ax.set_ylim([0, 1.2])
    ax.set_zlim([0, 1.2])
    ax.view_init(elev=30, azim=45)
    #ax.view_init(elev=1, azim=0)
    return ax


def visualize_base_trajectories(trajectory):
    plt.plot(trajectory[:,0], label="A")
    plt.plot(trajectory[:,1], label="C")
    plt.plot(trajectory[:,2], label="G")
    plt.plot(trajectory[:,3], label="T")
    plt.title("Optimization Trajectory of each base component")
    plt.xlabel("Iteration")
    plt.legend()
    plt.show()
    plt.close()