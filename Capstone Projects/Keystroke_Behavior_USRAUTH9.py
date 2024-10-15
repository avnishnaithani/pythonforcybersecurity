import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import time

def get_keystroke_features(text_to_type):
    print(f"\nPlease type the following text:")
    print(f"'{text_to_type}'")
    print("Press Enter when you're ready to start typing.")
    input()
    
    print("\nStart typing now:")
    start_time = time.time()
    typed_text = input()
    end_time = time.time()
    
    if typed_text != text_to_type:
        print("The typed text doesn't match the required text. Please try again.")
        return None

    total_time = end_time - start_time
    
    # Calculate features
    avg_time_per_char = total_time / len(text_to_type)
    typing_speed = len(text_to_type) / total_time  # characters per second
    
    # Additional features
    first_half_time = total_time / 2
    second_half_time = total_time - first_half_time
    time_ratio = first_half_time / second_half_time
    
    return [avg_time_per_char, typing_speed, time_ratio]

def train_model(text_to_type, num_samples=7):
    print("\nTraining phase:")
    features = []
    
    while len(features) < num_samples:
        print(f"\nSample {len(features) + 1}/{num_samples}")
        sample_features = get_keystroke_features(text_to_type)
        if sample_features:
            features.append(sample_features)
        else:
            print("Sample discarded. Please try again.")
    
    X = np.array(features)
    y = np.ones(num_samples)  # All samples are from the authentic user
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = MLPClassifier(
        hidden_layer_sizes=(5, 3),
        max_iter=1000,
        alpha=0.01,
        solver='adam',
        learning_rate_init=0.001,
        activation='relu'
    )
    model.fit(X_scaled, y)
    
    return model, scaler

def authenticate(model, scaler, text_to_type, threshold):
    print("\nAuthentication phase:")
    features = get_keystroke_features(text_to_type)
    
    if features is None:
        return False, 0.0
    
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    probability = model.predict_proba(X_scaled)[0][1]
    
    # Debugging information
    print(f"Authentication features: {features}")
    print(f"Scaled features: {X_scaled}")
    print(f"Authentication probability: {probability}")
    print(f"Threshold: {threshold}")
    
    return probability >= threshold, probability

def main():
    print("Keystroke Behavior Authentication System")
    print("========================================")
    
    text_to_type = "The quick brown fox jumps over the lazy dog"
    model, scaler = train_model(text_to_type)
    
    # Set a very low threshold
    threshold = 0.03
    
    while True:
        is_authentic, probability = authenticate(model, scaler, text_to_type, threshold)
        
        if is_authentic:
            print(f"Authentication successful! Confidence: {probability:.2f}")
        else:
            print(f"Authentication failed. Confidence: {probability:.2f}")
        
        choice = input("\nDo you want to try again? (y/n): ")
        if choice.lower() != 'y':
            break

    print("Thank you for using the Keystroke Behavior Authentication System!")

if __name__ == "__main__":
    main()
