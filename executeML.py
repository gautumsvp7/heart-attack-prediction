import pickle
import numpy as np

def load_model(model_path='stackedModel.pkl'):
    """Load the pre-trained machine learning model."""
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def predict(input_features, model):
    """
    Predict using the loaded model.
    
    Parameters:
    - input_features: List of features in the order:
      ['age', 'gender', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi']
    - model: Loaded machine learning model
    
    Returns:
    - Prediction result
    """
    input_array = np.array(input_features).reshape(1, -1)
    return model.predict(input_array)

if __name__ == "__main__":
    # Example input
    input_data = [50, 1, 120, 80, 1, 1, 0, 0, 1, 25.0]  # Replace with actual input
    model = load_model()
    result = predict(input_data, model)
    print(f"Prediction: {result[0]}")
