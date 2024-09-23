import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def create_model(data): 
    # Remove 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
    
    X = data.drop(['LUNG_CANCER'], axis=1)
    y = data['LUNG_CANCER']

    # Store column names before scaling
    feature_names = X.columns.tolist()

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))

    return model, scaler, feature_names  # Return feature_names instead of X.columns

def get_clean_data():
    data_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_lung_cancer.csv')
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found at {data_file}")
    data = pd.read_csv(data_file)
    return data

def main():
    data = get_clean_data()

    model, scaler, feature_names = create_model(data)

    model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'lung_cancer_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    feature_names_path = os.path.join(model_dir, 'feature_names.pkl')

    print(f"Saving model to: {model_path}")
    print(f"Saving scaler to: {scaler_path}")
    print(f"Saving feature names to: {feature_names_path}")

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(feature_names_path, 'wb') as f:
        pickle.dump(feature_names, f)

    print("Files saved successfully.")

if __name__ == '__main__':
    main()