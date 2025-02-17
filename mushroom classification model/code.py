# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"🔥 Model Accuracy: {accuracy * 100:.2f}%")
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# Function to predict new data
def predict_mushroom(features_dict):
    input_data = np.array([label_encoders[col].transform([features_dict[col]])[0] for col in X.columns]).reshape(1, -1)
    prediction = model.predict(input_data)
    return "Edible 🍄" if prediction[0] == 0 else "Poisonous ☠️"

# Example prediction
sample_mushroom = {
    "cap-shape": "x",
    "cap-surface": "s",
    "cap-color": "n",
    "bruises": "t",
    "odor": "p",
    "gill-attachment": "f",
    "gill-spacing": "c",
    "gill-size": "b",
    "gill-color": "k",
    "stalk-shape": "e",
    "stalk-root": "c",
    "stalk-surface-above-ring": "s",
    "stalk-surface-below-ring": "s",
    "stalk-color-above-ring": "w",
    "stalk-color-below-ring": "w",
    "veil-type": "p",
    "veil-color": "w",
    "ring-number": "o",
    "ring-type": "p",
    "spore-print-color": "k",
    "population": "s",
    "habitat": "u"
}
print("🧐 Sample Prediction:", predict_mushroom(sample_mushroom))
import joblib

# Save model
joblib.dump(model, "mushroom_classifier.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("✅ Model saved as mushroom_classifier.pkl")
