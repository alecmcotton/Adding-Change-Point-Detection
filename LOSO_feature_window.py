import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

class GestureDataProcessor:
    def __init__(self, window_size=50):
        self.window_size = window_size
        
    def load_transcription(self, trans_path):
        """Load transcription file and return list of (start, end, label) tuples"""
        if not os.path.exists(trans_path):
            return None
        
        transcriptions = []
        with open(trans_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) == 3:
                        start, end, label = int(parts[0]), int(parts[1]), parts[2].strip()
                        transcriptions.append((start, end, label))
        return transcriptions
    
    def get_labels_for_rows(self, transcriptions, num_rows):
        """Create array of labels for each row based on transcriptions"""
        labels = [None] * num_rows
        
        if transcriptions is None:
            return labels
        
        for start, end, label in transcriptions:
            for i in range(start, min(end + 1, num_rows)):
                if i < num_rows:
                    labels[i] = label
        
        return labels
    
    def extract_window_features(self, window_data):
        """Extract mean, variance, skewness, and kurtosis for each column in window"""
        features = []
        
        for col in window_data.columns:
            col_data = window_data[col].values
            features.append(np.mean(col_data))
            features.append(np.var(col_data))
            features.append(skew(col_data))
            features.append(kurtosis(col_data))
        
        return features
    
    def majority_vote_label(self, window_labels):
        """Get majority label in window, return None if no labels or tie"""
        valid_labels = [l for l in window_labels if l is not None]
        
        if not valid_labels:
            return None
        
        counter = Counter(valid_labels)
        most_common = counter.most_common(1)[0]
        
        return most_common[0]
    
    def process_file(self, kin_path, trans_path):
        """Process a single kinematics file with its transcription"""
        # Load kinematics data
        kin_data = pd.read_csv(kin_path, header=None)
        num_rows = len(kin_data)
        
        # Load transcriptions
        transcriptions = self.load_transcription(trans_path)
        
        if transcriptions is None:
            return None, None
        
        # Get labels for each row
        row_labels = self.get_labels_for_rows(transcriptions, num_rows)
        
        # Extract sliding window features
        X_windows = []
        y_windows = []
        
        for i in range(num_rows - self.window_size + 1):
            window_data = kin_data.iloc[i:i + self.window_size]
            window_labels = row_labels[i:i + self.window_size]
            
            # Get majority label for this window
            majority_label = self.majority_vote_label(window_labels)
            
            if majority_label is not None:
                features = self.extract_window_features(window_data)
                X_windows.append(features)
                y_windows.append(majority_label)
        
        return X_windows, y_windows
    
    def process_fold(self, fold_path, split='train'):
        """Process all files in a fold's train or test split"""
        kin_dir = os.path.join(fold_path, split, 'kinematics')
        trans_dir = os.path.join(fold_path, split, 'transcriptions')
        
        if not os.path.exists(kin_dir):
            return [], []
        
        X_all = []
        y_all = []
        
        kin_files = [f for f in os.listdir(kin_dir) if f.endswith('.csv')]
        
        for kin_file in kin_files:
            kin_path = os.path.join(kin_dir, kin_file)
            trans_path = os.path.join(trans_dir, kin_file)
            
            X_windows, y_windows = self.process_file(kin_path, trans_path)
            
            if X_windows is not None:
                X_all.extend(X_windows)
                y_all.extend(y_windows)
        
        return X_all, y_all


def find_activity_folders(base_dir):
    """Find all activity folders in the base directory"""
    activities = ['Knot_Tying', 'Needle_Passing', 'Suturing']
    found_activities = []
    
    for activity in activities:
        activity_path = os.path.join(base_dir, activity)
        if os.path.exists(activity_path):
            found_activities.append(activity)
    
    return found_activities


def find_user_and_fold_paths(base_dir, activity):
    """Find all User/Fold combinations for an activity"""
    activity_path = os.path.join(base_dir, activity)
    fold_paths = []
    
    if not os.path.exists(activity_path):
        return fold_paths
    
    # Find all User folders
    for item in os.listdir(activity_path):
        user_path = os.path.join(activity_path, item)
        if os.path.isdir(user_path) and item.startswith('User'):
            # Find all Fold folders
            for fold_item in os.listdir(user_path):
                fold_path = os.path.join(user_path, fold_item)
                if os.path.isdir(fold_path) and fold_item.startswith('Fold'):
                    fold_paths.append({
                        'path': fold_path,
                        'user': item,
                        'fold': fold_item
                    })
    
    return fold_paths


def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """Train and evaluate three models"""
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM (Linear)': SVC(kernel='linear', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for model_name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Store results
        results[model_name] = {
            'model': model,
            'y_true': y_test,
            'y_pred': y_pred
        }
    
    return results


def aggregate_predictions(all_fold_results):
    """Aggregate predictions across all folds"""
    aggregated = defaultdict(lambda: {'y_true': [], 'y_pred': []})
    
    for model_name in all_fold_results[0].keys():
        for fold_result in all_fold_results:
            aggregated[model_name]['y_true'].extend(fold_result[model_name]['y_true'])
            aggregated[model_name]['y_pred'].extend(fold_result[model_name]['y_pred'])
    
    return aggregated


def process_activity(base_dir, activity, processor):
    """Process all users and folds for a given activity"""
    print(f"\n{'='*80}")
    print(f"ACTIVITY: {activity}")
    print(f"{'='*80}")
    
    # Find all user/fold combinations
    fold_paths = find_user_and_fold_paths(base_dir, activity)
    
    if not fold_paths:
        print(f"No folds found for {activity}")
        return None
    
    print(f"Found {len(fold_paths)} folds across all users")
    
    all_fold_results = []
    
    # Process each fold
    for fold_info in fold_paths:
        fold_path = fold_info['path']
        user = fold_info['user']
        fold = fold_info['fold']
        
        print(f"\nProcessing {user}/{fold}...")
        
        # Load training data
        X_train, y_train = processor.process_fold(fold_path, 'train')
        
        # Load test data
        X_test, y_test = processor.process_fold(fold_path, 'test')
        
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"  Skipping {user}/{fold} - insufficient data")
            continue
        
        print(f"  Train: {len(X_train)} windows, Test: {len(X_test)} windows")
        
        # Convert to arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # Train and evaluate
        fold_results = train_and_evaluate_models(X_train, y_train, X_test, y_test)
        all_fold_results.append(fold_results)
    
    if not all_fold_results:
        print(f"No valid results for {activity}")
        return None
    
    # Aggregate results across all folds
    print(f"\n{'='*80}")
    print(f"AGGREGATED RESULTS FOR {activity}")
    print(f"{'='*80}")
    
    aggregated = aggregate_predictions(all_fold_results)
    
    # Print results for each model
    for model_name in sorted(aggregated.keys()):
        y_true = np.array(aggregated[model_name]['y_true'])
        y_pred = np.array(aggregated[model_name]['y_pred'])
        
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"\n{'-'*80}")
        print(f"Model: {model_name}")
        print(f"{'-'*80}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
    
    return aggregated


def main():
    # Configuration
    BASE_DIR = r'D:\AlecCotton\Combined Segmentation and Classification\LOSO'
    WINDOW_SIZE = 50
    
    print("Multi-Activity Gesture Classification Pipeline")
    print("="*80)
    
    # Initialize processor
    processor = GestureDataProcessor(window_size=WINDOW_SIZE)
    
    # Find all activities
    activities = find_activity_folders(BASE_DIR)
    
    if not activities:
        print(f"No activity folders found in {BASE_DIR}")
        return
    
    print(f"Found activities: {', '.join(activities)}")
    
    # Process each activity
    all_activity_results = {}
    
    for activity in activities:
        results = process_activity(BASE_DIR, activity, processor)
        if results is not None:
            all_activity_results[activity] = results
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    for activity in sorted(all_activity_results.keys()):
        print(f"\n{activity}:")
        for model_name in sorted(all_activity_results[activity].keys()):
            y_true = np.array(all_activity_results[activity][model_name]['y_true'])
            y_pred = np.array(all_activity_results[activity][model_name]['y_pred'])
            accuracy = accuracy_score(y_true, y_pred)
            print(f"  {model_name}: {accuracy:.4f}")


if __name__ == "__main__":
    main()