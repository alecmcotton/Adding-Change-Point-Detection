import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import beta
import warnings
from sklearn.metrics import confusion_matrix, classification_report
warnings.filterwarnings('ignore')

ROOT = r"D:\AlecCotton"

def LOUO_loading(ROOT):
    data = {}
    dir_path = os.path.join(ROOT, "Gesture Classification", "LOUO")
    for activity in os.listdir(dir_path):
        activity_path = os.path.join(dir_path, activity)
        if not os.path.isdir(activity_path):
            continue
        data[activity] = {}
        for fold in os.listdir(activity_path):
            fold_path = os.path.join(activity_path, fold, "test")
            if not os.path.isdir(fold_path):
                continue
            data[activity][fold] = {}
            for file in os.listdir(fold_path):
                file_path = os.path.join(fold_path, file)
                gesture_label = label(file)
                if gesture_label == "":
                    continue
                df = pd.read_csv(file_path)
                if gesture_label not in data[activity][fold]:
                    data[activity][fold][gesture_label] = [df]
                else:
                    data[activity][fold][gesture_label].append(df)
    return data

def label(filename):
    parts = filename.rsplit("_", 2)
    if len(parts) < 3:
        return "" 
    return parts[1]

def feature_preprocessing(data):
    processed_data = {}
    for activity, folds in data.items():
        processed_data[activity] = {}
        for fold, gestures in folds.items():
            processed_data[activity][fold] = {}
            for gesture, df_list in gestures.items():
                processed_data[activity][fold][gesture] = [
                    extract_features(df) for df in df_list
                ]
    return processed_data

def extract_features(df):
    if df.empty:
        return np.zeros(df.shape[1] * 4)
    means = df.mean()
    variances = df.var()
    skews = df.skew()
    kurtoses = df.kurtosis()
    features = pd.concat([means, variances, skews, kurtoses], axis=0)
    return features.values


def LOUO_clf(processed_data, classifiers):
    results = {}
    for activity, folds in processed_data.items():
        print(f"Processing activity: {activity}")
        results[activity] = {}
        all_labels = sorted({
            gesture for fold_data in folds.values() for gesture in fold_data.keys()
        })
        
        for model_name, clf_func in classifiers.items():
            print(f"  Training {model_name}...")
            confusion_matrices = []
            all_y_true, all_y_pred = [], []
            fold_names = list(folds.keys())
            
            for test_fold in fold_names:
                X_train, y_train, X_test, y_test = [], [], [], []
                for fold, gestures in folds.items():
                    for gesture, vecs in gestures.items():
                        if not vecs:
                            continue
                        if fold == test_fold:
                            X_test.extend(vecs)
                            y_test.extend([gesture] * len(vecs))
                        else:
                            X_train.extend(vecs)
                            y_train.extend([gesture] * len(vecs))

                X_train = np.array([x for x in X_train if x.size > 0])
                y_train = np.array([y for x, y in zip(X_train, y_train) if x.size > 0])
                X_test = np.array([x for x in X_test if x.size > 0])
                y_test = np.array([y for x, y in zip(X_test, y_test) if x.size > 0])

                if X_train.size == 0 or X_test.size == 0:
                    print(f"    Skipping fold {test_fold} for model {model_name} due to empty data")
                    continue

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                try:
                    clf = clf_func()
                    clf.fit(X_train_scaled, y_train)
                    y_pred = clf.predict(X_test_scaled)
                    cm = confusion_matrix(y_test, y_pred, labels=all_labels)
                    confusion_matrices.append(cm)
                    
                    all_y_true.extend(y_test)
                    all_y_pred.extend(y_pred)

                except Exception as e:
                    print(f"    Error with {model_name} on fold {test_fold}: {e}")
                    continue

            results[activity][model_name] = {
                "conf_matrices": confusion_matrices,
                "y_true": all_y_true,
                "y_pred": all_y_pred,
                "labels": all_labels
            }
    return results

def evaluate_results(results):
    metrics = {}
    for activity, models in results.items():
        metrics[activity] = {}
        for model, data in models.items():
            cms = [cm for cm in data["conf_matrices"] if isinstance(cm, np.ndarray)]
            if not cms:
                continue

            total_correct, total_samples = 0, 0
            per_class_acc, per_class_prec = [], []

            for cm in cms:
                correct = np.trace(cm)
                samples = np.sum(cm)
                if samples == 0:
                    continue
                total_correct += correct
                total_samples += samples

                with np.errstate(divide="ignore", invalid="ignore"):
                    acc_per_class = np.diag(cm) / cm.sum(axis=1)
                acc_per_class = acc_per_class[~np.isnan(acc_per_class)]
                if len(acc_per_class) > 0:
                    per_class_acc.append(np.mean(acc_per_class))

                with np.errstate(divide="ignore", invalid="ignore"):
                    prec_per_class = np.diag(cm) / cm.sum(axis=0)
                prec_per_class = prec_per_class[~np.isnan(prec_per_class)]
                if len(prec_per_class) > 0:
                    per_class_prec.append(np.mean(prec_per_class))

            micro_acc = total_correct / total_samples if total_samples > 0 else np.nan
            macro_acc_mean = np.mean(per_class_acc) if per_class_acc else np.nan
            macro_acc_std = np.std(per_class_acc) if per_class_acc else np.nan
            prec_mean = np.mean(per_class_prec) if per_class_prec else np.nan
            prec_std = np.std(per_class_prec) if per_class_prec else np.nan

            successes = int(sum(np.trace(cm) for cm in cms))
            trials = int(sum(np.sum(cm) for cm in cms))
            alpha, beta_param = successes + 1, (trials - successes) + 1
            dist = beta(alpha, beta_param)
            ci_low, ci_high = dist.interval(0.95)

            metrics[activity][model] = {
                "micro_accuracy": micro_acc,
                "macro_accuracy_mean": macro_acc_mean,
                "macro_accuracy_std": macro_acc_std,
                "precision_mean": prec_mean,
                "precision_std": prec_std,
                "beta_mean": dist.mean(),
                "beta_std": dist.std(),
                "beta_CI_95": (ci_low, ci_high),
                "y_true": data["y_true"],
                "y_pred": data["y_pred"],
                "labels": data["labels"]
            }

    return metrics

def find_top_performers(all_metrics, top_n=3):
    """Find top N performers across all activities based on micro accuracy"""
    model_scores = {}
    
    for activity, models in all_metrics.items():
        for model_name, metrics in models.items():
            if model_name not in model_scores:
                model_scores[model_name] = []
            model_scores[model_name].append(metrics['micro_accuracy'])
    
    model_averages = {}
    for model_name, scores in model_scores.items():
        valid_scores = [s for s in scores if not np.isnan(s)]
        if valid_scores:
            model_averages[model_name] = np.mean(valid_scores)
        else:
            model_averages[model_name] = 0.0
    
    sorted_models = sorted(model_averages.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_models[:top_n], model_averages

def main(ROOT):
    print("Loading data...")
    raw_data = LOUO_loading(ROOT)
    processed_data = feature_preprocessing(raw_data)
    
    feature_classifiers = {
        "SVM_RBF": lambda: SVC(kernel='rbf', C=1.0, gamma='scale'),
        "SVM_Linear": lambda: SVC(kernel='linear', C=1.0),
        "SVM_Poly": lambda: SVC(kernel='poly', degree=3, C=1.0),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
        "ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
        "AdaBoost": lambda: AdaBoostClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=42),
        "RidgeClassifier": lambda: RidgeClassifier(random_state=42),
        "KNN_5": lambda: KNeighborsClassifier(n_neighbors=5),
        "KNN_3": lambda: KNeighborsClassifier(n_neighbors=3),
        "KNN_7": lambda: KNeighborsClassifier(n_neighbors=7),
        "DecisionTree": lambda: DecisionTreeClassifier(random_state=42),
        "NaiveBayes": lambda: GaussianNB(),
        "MLP_Small": lambda: MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42),
        "MLP_Large": lambda: MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        "LinearDiscriminant": lambda: LinearDiscriminantAnalysis(),
        "QuadraticDiscriminant": lambda: QuadraticDiscriminantAnalysis(),
    }
    
    print("Running feature-based classifiers...")
    feature_results = LOUO_clf(processed_data, feature_classifiers)
    feature_metrics = evaluate_results(feature_results)
    
    output_file = os.path.join(ROOT, "LOUO_feature_clf_results.txt")
    with open(output_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("LOUO FEATURE-BASED CLASSIFIER RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for activity, models in feature_metrics.items():
            f.write(f"{activity.upper()}:\n")
            f.write("-"*50 + "\n")
            for model_name, metrics in models.items():
                f.write(
                    f"{model_name:20}: "
                    f"Micro Acc: {metrics['micro_accuracy']:.3f} | "
                    f"Macro Acc: {metrics['macro_accuracy_mean']:.3f} ± {metrics['macro_accuracy_std']:.3f} | "
                    f"Precision: {metrics['precision_mean']:.3f} ± {metrics['precision_std']:.3f} | "
                    f"Beta Mean: {metrics['beta_mean']:.3f} ± {metrics['beta_std']:.3f} | "
                    f"95% CI: [{metrics['beta_CI_95'][0]:.3f}, {metrics['beta_CI_95'][1]:.3f}]\n"
                )
                
                # --- add classification report ---
                if metrics["y_true"] and metrics["y_pred"]:
                    report = classification_report(
                        metrics["y_true"], metrics["y_pred"], labels=metrics["labels"], zero_division=0
                    )
                    f.write("\nClassification Report:\n")
                    f.write(report + "\n")
                
            f.write("\n")


if __name__ == "__main__":
    main(ROOT)