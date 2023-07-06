import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    sns.heatmap(arr, annot = True)
    plt.show()