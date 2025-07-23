import numpy as np
from scipy.io import arff


def padding_varying_length(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j, :][np.isnan(data[i, j, :])] = 0
    return data


def load_UCR(Path='../../archives/UCR_UEA/Multivariate_arff/', folder='Cricket'):
    import numpy as np
    from sklearn.model_selection import train_test_split

    X = np.load("C:/Users/hp/Downloads/FormerTime-main/FormerTime-main/data_features.npy")      # (N, C, T)
    y = np.load("C:/Users/hp/Downloads/FormerTime-main/FormerTime-main/data_targets.npy")       # (N,)
    X = X.transpose(0, 2, 1)              # → (N, T, C)
    MAX_TIME_STEPS = 6000
    X = X[:, :MAX_TIME_STEPS, :]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    return (X_train, y_train), (X_test, y_test)
