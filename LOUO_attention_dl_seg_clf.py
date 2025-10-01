import os
import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score, precision_recall_fscore_support
from scipy.stats import beta as sp_beta

from LOSO_dl_seg_clf import GestureDataset, train_model, evaluate_model, fit_beta_from_samples

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ChangePointAttention(nn.Module):
    """
    Attention mechanism that focuses on change points between gestures.
    Computes attention weights based on temporal differences in the sequence.
    """
    def __init__(self, hidden_size):
        super(ChangePointAttention, self).__init__()
        self.hidden_size = hidden_size
        
        # Networks to compute change point scores
        self.change_detector = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Attention scoring
        self.attention_weights = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
        Returns:
            context: (batch, hidden_size) - weighted sum of hidden states
            attention_weights: (batch, seq_len) - attention distribution
            change_scores: (batch, seq_len-1) - change point scores
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute temporal differences to detect changes
        if seq_len > 1:
            diff = hidden_states[:, 1:, :] - hidden_states[:, :-1, :]
            
            # Concatenate consecutive states to detect transitions
            paired = torch.cat([hidden_states[:, :-1, :], hidden_states[:, 1:, :]], dim=-1)
            change_scores = self.change_detector(paired).squeeze(-1)  # (batch, seq_len-1)
            
            # Pad change scores to match sequence length
            change_scores_padded = F.pad(change_scores, (0, 1), value=0)  # (batch, seq_len)
        else:
            change_scores_padded = torch.zeros(batch_size, seq_len, device=hidden_states.device)
            change_scores = change_scores_padded
        
        # Compute attention scores combining change detection and content
        content_scores = self.attention_weights(hidden_states).squeeze(-1)  # (batch, seq_len)
        
        # Combine change point scores with content attention
        combined_scores = content_scores + change_scores_padded
        attention_weights = F.softmax(combined_scores, dim=1)  # (batch, seq_len)
        
        # Compute weighted context vector
        context = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1)  # (batch, hidden_size)
        
        return context, attention_weights, change_scores


class GRUModelWithAttention(nn.Module):
    """GRU with change point attention"""
    def __init__(self, input_size=76, hidden_size=128, num_layers=2, num_classes=16, dropout=0.3):
        super(GRUModelWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0,
                         bidirectional=False)
        
        self.attention = ChangePointAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Classifier uses both sequential output and attention context
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, hidden = self.gru(x)  # out: (batch, seq_len, hidden_size)
        
        # Apply change point attention
        context, attn_weights, change_scores = self.attention(out)
        
        # Combine attention context with each timestep
        context_expanded = context.unsqueeze(1).expand(-1, out.size(1), -1)
        combined = torch.cat([out, context_expanded], dim=-1)
        
        combined = self.dropout(combined)
        logits = self.fc(combined)  # (batch, seq_len, num_classes)
        
        return logits


class LSTMModelWithAttention(nn.Module):
    """LSTM with change point attention"""
    def __init__(self, input_size=76, hidden_size=128, num_layers=2, num_classes=16, dropout=0.3):
        super(LSTMModelWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=False)
        
        self.attention = ChangePointAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        
        # Apply change point attention
        context, attn_weights, change_scores = self.attention(out)
        
        # Combine attention context with each timestep
        context_expanded = context.unsqueeze(1).expand(-1, out.size(1), -1)
        combined = torch.cat([out, context_expanded], dim=-1)
        
        combined = self.dropout(combined)
        logits = self.fc(combined)
        
        return logits


class TCNBlock(nn.Module):
    """Temporal Convolutional Block"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.3):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]  # Trim padding
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = out[:, :, :x.size(2)]  # Trim padding
        out = self.relu2(out)
        out = self.dropout2(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModelWithAttention(nn.Module):
    """TCN with change point attention"""
    def __init__(self, input_size=76, hidden_size=128, num_layers=4, num_classes=16, 
                 kernel_size=3, dropout=0.3):
        super(TCNModelWithAttention, self).__init__()
        
        layers = []
        num_channels = [hidden_size] * num_layers
        
        for i in range(num_layers):
            in_ch = input_size if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        
        self.network = nn.Sequential(*layers)
        self.attention = ChangePointAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x_transposed = x.transpose(1, 2)  # (batch, input_size, seq_len)
        out = self.network(x_transposed)  # (batch, hidden_size, seq_len)
        out = out.transpose(1, 2)  # (batch, seq_len, hidden_size)
        
        # Apply change point attention
        context, attn_weights, change_scores = self.attention(out)
        
        # Combine attention context with each timestep
        context_expanded = context.unsqueeze(1).expand(-1, out.size(1), -1)
        combined = torch.cat([out, context_expanded], dim=-1)
        
        combined = self.dropout(combined)
        logits = self.fc(combined)
        
        return logits


def run_full_experiment_louo(root_base, output_base, sequence_length=128, overlap=0.5,
                             batch_size=32, num_epochs=30, lr=1e-3, expected_cols=76):
    """
    root_base: top folder containing activity folders (Needle_Passing, Knot_Tying, Suturing)
    structure expected:
    <root_base>/<activity>/<user>/{train,test}/kinematics/*.csv and transcriptions/*.csv
    
    Uses models WITH change point attention mechanisms.
    """
    root_base = Path(root_base)
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    activities = [p for p in root_base.iterdir() if p.is_dir()]
    overall_summary = {}

    for activity_path in activities:
        activity = activity_path.name
        print(f"\n=== Activity: {activity} ===")
        activity_out = output_base / activity
        activity_out.mkdir(exist_ok=True, parents=True)

        model_user_metrics = defaultdict(list) 
        pooled_predictions = defaultdict(lambda: {'all_preds': [], 'all_targets': []})

        users = [u for u in activity_path.iterdir() if u.is_dir()]
        if not users:
            print(f"No user folders under {activity_path}, skipping")
            continue

        for user in users:
            user_name = user.name
            print(f"\n--- Processing user: {user_name} ---")
            user_out = activity_out / user_name
            user_out.mkdir(exist_ok=True, parents=True)

            train_dir = user / 'train'
            test_dir = user / 'test'
            if not train_dir.exists() or not test_dir.exists():
                print(f"Missing train/test in {user}. Skipping user.")
                continue

            try:
                train_ds = GestureDataset(str(train_dir), sequence_length=sequence_length,
                                          overlap=overlap, expected_cols=expected_cols, scaler=None)
            except Exception as e:
                print(f"Could not create train dataset for {user_name}: {e}")
                continue

            scaler = train_ds.scaler
            try:
                test_ds = GestureDataset(str(test_dir), sequence_length=sequence_length,
                                         overlap=overlap, expected_cols=expected_cols, scaler=scaler)
            except Exception as e:
                print(f"Could not create test dataset for {user_name}: {e}")
                continue

            train_size = int(0.8 * len(train_ds))
            val_size = len(train_ds) - train_size
            try:
                train_subset, val_subset = torch.utils.data.random_split(train_ds, [train_size, val_size])
            except Exception as e:
                print(f"Split error: {e}")
                continue

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

            # Models with change point attention
            models = {
                'GRU_Attention': GRUModelWithAttention(input_size=expected_cols),
                'LSTM_Attention': LSTMModelWithAttention(input_size=expected_cols),
                'TCN_Attention': TCNModelWithAttention(input_size=expected_cols)
            }

            for model_name, model in models.items():
                print(f"Training {model_name} on {user_name}")
                try:
                    trained_model, history = train_model(model, train_loader, val_loader,
                                                         num_epochs=num_epochs, lr=lr)
                except Exception as e:
                    print(f"Training failed for {model_name} on {user_name}: {e}")
                    continue

                model_path = user_out / f"{model_name.lower()}_best.pth"
                torch.save(trained_model.state_dict(), str(model_path))

                metrics = evaluate_model(trained_model, test_loader, model_name, str(user_out))
                if metrics is None:
                    print(f"No test data for {user_name}/{model_name}, skipping.")
                    continue

                # Collect predictions for pooled analysis
                trained_model.eval()
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)
                        outputs = trained_model(x)
                        out_flat = outputs.reshape(-1, outputs.size(-1))
                        tgt = y.reshape(-1)
                        _, pred = torch.max(out_flat, 1)
                        pooled_predictions[model_name]['all_preds'].extend(pred.cpu().numpy())
                        pooled_predictions[model_name]['all_targets'].extend(tgt.cpu().numpy())

                # Extract metrics from evaluate_model
                if 'accuracy' in metrics:
                    accuracy = metrics['accuracy']
                    precision = metrics['precision']
                    recall = metrics['recall']
                    f1_score = metrics['f1_score']
                    beta_ci = metrics.get('beta_CI_95', (None, None))
                    class_report_str = metrics.get('classification_report_str', '')
                else:
                    # Fallback for older evaluate_model that returns micro_acc
                    accuracy = metrics.get('micro_acc', 0.0)
                    precision = metrics.get('macro_precision', 0.0)
                    recall = metrics.get('macro_recall', 0.0)
                    f1_score = 0.0
                    beta_ci = (None, None)
                    class_report_str = str(metrics.get('classification_report', ''))

                model_user_metrics[model_name].append({
                    'user': user_name,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'beta_CI_95': beta_ci,
                    'present_labels': metrics.get('present_labels', []),
                    'classification_report': class_report_str
                })

        # Generate activity-level summary with pooled metrics
        activity_summary = {}
        activity_report_path = activity_out / 'activity_classification_reports.txt'
        
        with open(activity_report_path, 'w') as report_file:
            report_file.write("="*80 + "\n")
            report_file.write(f"ACTIVITY: {activity.upper()} - LOUO (WITH CHANGE POINT ATTENTION)\n")
            report_file.write("="*80 + "\n\n")
            
            for model_name, user_list in model_user_metrics.items():
                if len(user_list) == 0:
                    continue
                    
                report_file.write(f"\n{'-'*80}\n")
                report_file.write(f"MODEL: {model_name}\n")
                report_file.write(f"{'-'*80}\n\n")
                
                # Pooled classification metrics
                pooled_data = pooled_predictions[model_name]
                all_preds = np.array(pooled_data['all_preds'])
                all_targets = np.array(pooled_data['all_targets'])
                
                if len(all_targets) > 0:
                    present_labels = np.unique(np.concatenate([all_targets, all_preds]))
                    present_labels_sorted = np.sort(present_labels).tolist()
                    
                    target_names = []
                    for lab in present_labels_sorted:
                        if lab == 0:
                            target_names.append('Background')
                        else:
                            target_names.append(f'G{lab}')
                    
                    pooled_accuracy = accuracy_score(all_targets, all_preds)
                    pooled_precision, pooled_recall, pooled_f1, _ = precision_recall_fscore_support(
                        all_targets, all_preds, average='weighted', zero_division=0
                    )
                    
                    # Beta CI
                    successes = int(np.sum(all_preds == all_targets))
                    trials = len(all_targets)
                    alpha, beta_param = successes + 1, (trials - successes) + 1
                    dist = sp_beta(alpha, beta_param)
                    ci_low, ci_high = dist.interval(0.95)
                    
                    report_file.write(f"Overall Metrics (Pooled across all users):\n")
                    report_file.write(f"  Accuracy:  {pooled_accuracy:.4f}\n")
                    report_file.write(f"  Precision: {pooled_precision:.4f}\n")
                    report_file.write(f"  Recall:    {pooled_recall:.4f}\n")
                    report_file.write(f"  F1-Score:  {pooled_f1:.4f}\n")
                    report_file.write(f"  95% CI:    [{ci_low:.4f}, {ci_high:.4f}]\n\n")
                    
                    # Full classification report
                    class_report = classification_report(all_targets, all_preds, 
                                                        labels=present_labels_sorted,
                                                        target_names=target_names, 
                                                        zero_division=0)
                    report_file.write(f"Classification Report:\n")
                    report_file.write(f"{class_report}\n")
                    
                    # Confusion matrix
                    cm = confusion_matrix(all_targets, all_preds, labels=present_labels_sorted)
                    report_file.write(f"\nConfusion Matrix:\n")
                    report_file.write(f"{cm}\n\n")
                    
                    # Per-user summary statistics
                    report_file.write(f"Per-User Statistics:\n")
                    accuracies = [u['accuracy'] for u in user_list]
                    precisions = [u['precision'] for u in user_list]
                    recalls = [u['recall'] for u in user_list]
                    f1_scores = [u['f1_score'] for u in user_list]
                    
                    report_file.write(f"  Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n")
                    report_file.write(f"  Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}\n")
                    report_file.write(f"  Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}\n")
                    report_file.write(f"  F1-Score:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}\n\n")
                    
                    activity_summary[model_name] = {
                        'num_users': len(user_list),
                        'pooled_accuracy': float(pooled_accuracy),
                        'pooled_precision': float(pooled_precision),
                        'pooled_recall': float(pooled_recall),
                        'pooled_f1': float(pooled_f1),
                        'beta_CI_95': (float(ci_low), float(ci_high)),
                        'per_user_mean_accuracy': float(np.mean(accuracies)),
                        'per_user_std_accuracy': float(np.std(accuracies)),
                        'users_detail': user_list
                    }
                
                # Save per-user results to CSV
                df_rows = []
                for u in user_list:
                    df_rows.append({
                        'user': u['user'],
                        'accuracy': u['accuracy'],
                        'precision': u['precision'],
                        'recall': u['recall'],
                        'f1_score': u['f1_score'],
                        'ci_low': u['beta_CI_95'][0] if u['beta_CI_95'][0] is not None else np.nan,
                        'ci_high': u['beta_CI_95'][1] if u['beta_CI_95'][1] is not None else np.nan,
                        'present_labels': ",".join(map(str, u['present_labels']))
                    })
                pd.DataFrame(df_rows).to_csv(activity_out / f"{model_name}_per_user_results.csv", index=False)

        overall_summary[activity] = activity_summary

        with open(activity_out / 'activity_summary.json', 'w') as fh:
            json.dump(activity_summary, fh, indent=2)

    # Overall summary across activities
    with open(output_base / 'overall_summary.json', 'w') as fh:
        json.dump(overall_summary, fh, indent=2)
    
    # Create overall ranking report
    with open(output_base / 'overall_ranking.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("OVERALL PERFORMANCE RANKING ACROSS ALL ACTIVITIES (LOUO + ATTENTION)\n")
        f.write("="*80 + "\n\n")
        
        model_avg_scores = defaultdict(list)
        for activity, models in overall_summary.items():
            for model_name, metrics in models.items():
                model_avg_scores[model_name].append(metrics['pooled_accuracy'])
        
        model_avgs = {m: np.mean(scores) for m, scores in model_avg_scores.items()}
        sorted_models = sorted(model_avgs.items(), key=lambda x: x[1], reverse=True)
        
        for i, (model_name, avg_score) in enumerate(sorted_models, 1):
            f.write(f"{i}. {model_name:20}: {avg_score:.4f}\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("DETAILED BREAKDOWN BY ACTIVITY\n")
        f.write(f"{'='*80}\n\n")
        
        for model_name, avg_score in sorted_models:
            f.write(f"\n{model_name} (Average: {avg_score:.4f}):\n")
            f.write(f"{'-'*60}\n")
            for activity, models in overall_summary.items():
                if model_name in models:
                    m = models[model_name]
                    f.write(f"  {activity:20}: Acc={m['pooled_accuracy']:.4f}, "
                           f"P={m['pooled_precision']:.4f}, R={m['pooled_recall']:.4f}, "
                           f"F1={m['pooled_f1']:.4f}\n")
            f.write("\n")

    print("\nLOUO Experiment with Change Point Attention finished. Results written to:", output_base)
    return overall_summary


if __name__ == "__main__":
    ROOT = r"D:\AlecCotton\Combined Segmentation and Classification\LOUO"
    OUT = r"D:\AlecCotton\dl_seg_clf_louo_attention"

    SEQ_LEN = 128
    OVERLAP = 0.5
    BATCH = 32
    EPOCHS = 30
    LR = 1e-3
    EXPECTED_COLS = 76

    run_full_experiment_louo(ROOT, OUT, sequence_length=SEQ_LEN, overlap=OVERLAP,
                             batch_size=BATCH, num_epochs=EPOCHS, lr=LR, expected_cols=EXPECTED_COLS)
