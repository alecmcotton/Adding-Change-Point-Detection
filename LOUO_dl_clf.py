import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report
from scipy.stats import beta
import warnings
warnings.filterwarnings('ignore')

ROOT = r"D:\AlecCotton"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Dataset ----------------
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), self.labels[idx]

def collate_fn(batch):
    # Pads sequences to the max length in batch
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    padded = torch.zeros(len(sequences), max_len, sequences[0].shape[1])
    for i, seq in enumerate(sequences):
        padded[i, :len(seq), :] = seq
    return padded.to(DEVICE), torch.tensor(labels, dtype=torch.long).to(DEVICE), lengths

# ---------------- Sequence Models (Unidirectional) ----------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed)
        out = self.fc(h_n[-1])
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=10):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, h_n = self.gru(packed)
        out = self.fc(h_n[-1])
        return out

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x[:, :, :x.size(2) - (self.conv.padding[0])]  # remove extra padding

class TCNModel(nn.Module):
    def __init__(self, input_size, num_channels=[64,64], kernel_size=3, num_classes=10):
        super().__init__()
        layers = []
        in_ch = input_size
        for out_ch in num_channels:
            layers.append(TCNBlock(in_ch, out_ch, kernel_size))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x, lengths=None):
        x = x.permute(0,2,1)  # B, C, T
        x = self.tcn(x)
        x = x.mean(dim=2)  # global average pooling over time
        return self.fc(x)

# ---------------- Data Loading ----------------
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
                # standardize each column
                scaler = StandardScaler()
                df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
                if gesture_label not in data[activity][fold]:
                    data[activity][fold][gesture_label] = [df_scaled.values]
                else:
                    data[activity][fold][gesture_label].append(df_scaled.values)
    return data

def label(filename):
    parts = filename.rsplit("_", 2)
    if len(parts) < 3:
        return "" 
    return parts[1]

# ---------------- Training / Evaluation ----------------
def train_sequence_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for X, y, lengths in train_loader:
            optimizer.zero_grad()
            outputs = model(X, lengths)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

def predict_sequence_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y, lengths in test_loader:
            outputs = model(X, lengths)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

# ---------------- Main LOUO Sequence Evaluation ----------------
def LOUO_sequence_clf(data, model_type="LSTM", epochs=10, batch_size=16):
    results = {}
    for activity, folds in data.items():
        print(f"Processing activity: {activity}")
        results[activity] = {}
        all_labels = sorted({gesture for fold_data in folds.values() for gesture in fold_data.keys()})
        label_to_idx = {label:i for i,label in enumerate(all_labels)}
        idx_to_label = {i:label for label, i in label_to_idx.items()}
        
        all_y_true = []
        all_y_pred = []
        
        for fold_name in folds.keys():
            X_train, y_train, X_test, y_test = [], [], [], []
            for fold, gestures in folds.items():
                for gesture, sequences in gestures.items():
                    idx_label = label_to_idx[gesture]
                    if fold == fold_name:
                        X_test.extend(sequences)
                        y_test.extend([idx_label]*len(sequences))
                    else:
                        X_train.extend(sequences)
                        y_train.extend([idx_label]*len(sequences))
            
            if not X_train or not X_test:
                continue
            
            # DataLoaders
            train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            test_loader = DataLoader(SequenceDataset(X_test, y_test), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            
            input_size = X_train[0].shape[1]
            num_classes = len(all_labels)
            
            # Initialize model (unidirectional)
            if model_type=="LSTM":
                model = LSTMModel(input_size, num_classes=num_classes).to(DEVICE)
            elif model_type=="GRU":
                model = GRUModel(input_size, num_classes=num_classes).to(DEVICE)
            elif model_type=="TCN":
                model = TCNModel(input_size, num_classes=num_classes).to(DEVICE)
            else:
                raise ValueError("Invalid model_type")
            
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            train_sequence_model(model, train_loader, criterion, optimizer, epochs=epochs)
            y_true, y_pred = predict_sequence_model(model, test_loader)
            
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
        
        # Store aggregated predictions and label mapping
        results[activity][model_type] = {
            'y_true': np.array(all_y_true),
            'y_pred': np.array(all_y_pred),
            'idx_to_label': idx_to_label,
            'num_classes': len(all_labels)
        }
    return results

# ---------------- Metrics Evaluation ----------------
def evaluate_results(results):
    metrics = {}
    for activity, models in results.items():
        metrics[activity] = {}
        for model, result_dict in models.items():
            y_true = result_dict['y_true']
            y_pred = result_dict['y_pred']
            idx_to_label = result_dict['idx_to_label']
            num_classes = result_dict['num_classes']
            
            if len(y_true) == 0:
                continue
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            
            # Per-class metrics
            precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
                y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
            
            # Beta distribution for confidence interval
            successes = int(np.sum(y_true == y_pred))
            trials = len(y_true)
            alpha, beta_param = successes + 1, (trials - successes) + 1
            dist = beta(alpha, beta_param)
            ci_low, ci_high = dist.interval(0.95)
            
            # Get classification report as string
            target_names = [idx_to_label[i] for i in range(num_classes)]
            class_report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
            
            metrics[activity][model] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": cm,
                "classification_report": class_report,
                "per_class_metrics": {
                    "precision": precision_per_class,
                    "recall": recall_per_class,
                    "f1_score": f1_per_class,
                    "support": support_per_class
                },
                "beta_mean": dist.mean(),
                "beta_std": dist.std(),
                "beta_CI_95": (ci_low, ci_high),
                "idx_to_label": idx_to_label
            }
    
    return metrics

# ---------------- Save Results ----------------
def save_metrics(metrics, output_file):
    # Calculate overall averages
    all_averages = {}
    for activity, models in metrics.items():
        for model_name, m in models.items():
            if model_name not in all_averages:
                all_averages[model_name] = []
            all_averages[model_name].append(m["accuracy"])
    
    for model_name in all_averages:
        all_averages[model_name] = np.mean(all_averages[model_name])
    
    sorted_models = sorted(all_averages.items(), key=lambda x: x[1], reverse=True)
    
    with open(output_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("LOUO SEQUENCE-BASED CLASSIFIER RESULTS (UNIDIRECTIONAL)\n")
        f.write("="*80 + "\n\n")
        
        for activity, models in metrics.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"{activity.upper()}\n")
            f.write(f"{'='*80}\n\n")
            
            for model_name, m in models.items():
                f.write(f"\n{'-'*80}\n")
                f.write(f"MODEL: {model_name}\n")
                f.write(f"{'-'*80}\n\n")
                
                # Overall metrics
                f.write(f"Overall Metrics:\n")
                f.write(f"  Accuracy:  {m['accuracy']:.4f}\n")
                f.write(f"  Precision: {m['precision']:.4f}\n")
                f.write(f"  Recall:    {m['recall']:.4f}\n")
                f.write(f"  F1-Score:  {m['f1_score']:.4f}\n")
                f.write(f"  95% CI:    [{m['beta_CI_95'][0]:.4f}, {m['beta_CI_95'][1]:.4f}]\n\n")
                
                # Classification report
                f.write(f"Classification Report:\n")
                f.write(f"{m['classification_report']}\n")
                
                # Confusion matrix
                f.write(f"\nConfusion Matrix:\n")
                f.write(f"{m['confusion_matrix']}\n")
                f.write("\n")
        
        # Overall ranking
        f.write(f"\n{'='*80}\n")
        f.write("OVERALL PERFORMANCE RANKING (Average Accuracy Across Activities)\n")
        f.write(f"{'='*80}\n\n")
        for i, (model_name, avg_score) in enumerate(sorted_models, 1):
            f.write(f"{i:2d}. {model_name:15}: {avg_score:.4f}\n")
        
        # Detailed breakdown for top performers
        f.write(f"\n{'='*80}\n")
        f.write("DETAILED BREAKDOWN BY ACTIVITY\n")
        f.write(f"{'='*80}\n\n")
        
        for model_name, avg_score in sorted_models:
            f.write(f"\n{model_name} (Average: {avg_score:.4f}):\n")
            f.write(f"{'-'*60}\n")
            for activity, models in metrics.items():
                if model_name in models:
                    m = models[model_name]
                    f.write(f"  {activity:15}: Acc={m['accuracy']:.4f}, "
                           f"P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1_score']:.4f}\n")
            f.write("\n")
    
    print(f"Results saved to {output_file}")

# ---------------- Main ----------------
if __name__ == "__main__":
    raw_data = LOUO_loading(ROOT)
    all_results = {}
    for model_type in ["LSTM","GRU","TCN"]:
        print(f"\nRunning {model_type} model...")
        results = LOUO_sequence_clf(raw_data, model_type=model_type, epochs=15, batch_size=16)
        all_results[model_type] = results
    
    # Merge results
    merged_metrics = {}
    for model_type, results in all_results.items():
        for activity, models in results.items():
            if activity not in merged_metrics:
                merged_metrics[activity] = {}
            merged_metrics[activity].update(models)
    
    # Evaluate and save
    metrics = evaluate_results(merged_metrics)
    save_metrics(metrics, os.path.join(ROOT, "LOUO_sequence_clf_results.txt"))