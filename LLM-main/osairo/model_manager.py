import numpy as np

def train_gaussian_process(X_train, y_train):
    """
    Train a Gaussian Process regression model using GPFlow.
    Lazy-import GPFlow to reduce overhead if not needed.
    """
    import gpflow
    kernel = gpflow.kernels.Matern52()
    model = gpflow.models.GPR(data=(X_train, y_train), kernel=kernel)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, variables=model.trainable_variables)
    print("Gaussian Process model trained.")
    return model

def train_neural_network(X_train, y_train, epochs=100):
    """
    Train a simple feedforward neural network using TensorFlow/Keras.
    Lazy-import TensorFlow if needed.
    """
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(y_train.shape[1])
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    print("Neural network model trained.")
    return model

def train_model(model_type: str, X_train, y_train):
    """
    Train a model based on model_type: 'gp' or 'nn', or load an existing model if desired.
    """
    model_type = model_type.lower().strip()
    load_option = input("Do you want to load an existing model? (yes/no) [no]: ").strip().lower()
    if load_option in ["yes", "y"]:
        path = input("Enter the file path to load the model: ").strip()
        try:
            import pickle
            with open(path, "rb") as f:
                model = pickle.load(f)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}. Proceeding to train a new model.")
    if model_type == 'gp':
        return train_gaussian_process(X_train, y_train)
    elif model_type == 'nn':
        return train_neural_network(X_train, y_train)
    else:
        raise ValueError("Unknown model type. Choose 'gp' or 'nn'.")

def get_most_uncertain_point(model, X_unlabeled, model_type='gp'):
    """
    Identify the most uncertain data point from the unlabeled set.
    For GP models, use the maximum predictive variance.
    For NN models, choose a random point as a placeholder.
    Returns (point, index, uncertainty).
    """
    if model_type == 'gp':
        mean, variance = model.predict_f(X_unlabeled)
        idx = int(np.argmax(variance))
        return X_unlabeled[idx], idx, float(variance[idx])
    elif model_type == 'nn':
        idx = np.random.choice(range(X_unlabeled.shape[0]))
        return X_unlabeled[idx], idx, None
    else:
        raise ValueError("Model type must be 'gp' or 'nn'.")
