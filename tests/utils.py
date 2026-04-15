import numpy as np

def create_synthetic_ehr(num_patients=10, num_features=8):
    print("Testing Utils")
    """
    Creates fake patient EHR data for testing.
    Returns X features and y labels.
    """
    np.random.seed(0)
    X = np.random.rand(num_patients, num_features)
    y = np.random.randint(0, 2, size=num_patients)
    
    return X, y


if __name__ == "__main__":
    X, y = create_synthetic_ehr() 
    print("X shape:", X.shape)  
    print("y shape:", y.shape)