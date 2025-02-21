import joblib

# Load the trained model, vectorizer, and label encoder
model = joblib.load("trained_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def predict_disease(symptoms):
    """
    Predict the disease based on the given symptoms.

    Args:
        symptoms (str): A string containing the symptoms.

    Returns:
        str: The predicted disease.
    """
    # Preprocess the input symptoms
    symptoms_vectorized = vectorizer.transform([symptoms]).toarray()

    # Predict the disease
    predicted_label = model.predict(symptoms_vectorized)

    # Decode the predicted label to get the disease name
    predicted_disease = label_encoder.inverse_transform(predicted_label)
    return predicted_disease[0]

def main():
    print("Welcome to the Disease Prediction System!")
    print("Enter your symptoms (comma-separated):")
    
    # Take input from the user
    input_symptoms = input("Symptoms: ").strip()

    # Ensure the input is not empty
    if not input_symptoms:
        print("Error: No symptoms entered. Please try again.")
        return

    # Predict the disease
    try:
        predicted_disease = predict_disease(input_symptoms)
        print(f"\nPredicted Disease: {predicted_disease}")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

# Run the program
if __name__ == "__main__":
    main()