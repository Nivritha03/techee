import flwr as fl
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data = pd.read_csv("train_data.csv")

# Preprocess the data
# Extract features (Symptoms) and labels (Name)
X = data["Symptoms"]  # Features (text data)
y = data["Name"]      # Labels (disease names)

# Convert text symptoms into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
X_train = vectorizer.fit_transform(X).toarray()

# Encode the labels (disease names) into numerical values
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y)

# Save the vectorizer and label encoder for future use
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

# Define the model
model = DecisionTreeClassifier()

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config=None):
        """Send model parameters to the server."""
        return []  # DecisionTreeClassifier doesn't have traditional parameters

    def set_parameters(self, parameters):
        """Receive global model parameters from the server (not needed for DecisionTree)."""
        pass

    def fit(self, parameters, config):
        """Train model locally on hospital data."""
        model.fit(X_train, y_train)
        joblib.dump(model, "trained_model.pkl")  # Save locally trained model
        return self.get_parameters(), len(X_train), {}

    def evaluate(self, parameters, config):
        """Dummy evaluation method required by Flower."""
        return 0.0, len(X_train), {}

# Start the client with the correct method
if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="192.168.137.194:8080",  # Replace with actual main node IP
        client=FlowerClient()
    )