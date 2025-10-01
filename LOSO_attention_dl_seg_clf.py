# full_experiment.py
import os
import sys
import glob
import json
import math
import shutil
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
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score, precision_recall_fscore_support
from scipy.stats import beta as sp_beta

from tqdm import tqdm

warnings.filterwarnings('ignore')

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class GestureDataset(Dataset):
    def __init__(self, data_folder, sequence_length=128, overlap=0.5, expected_cols=76, scaler=None):
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.expected_cols = expected_cols
        self.data = []
        self.labels = []
        self.scaler = scaler if scaler is not None else StandardScaler()
        self._load_data(data_folder)
        if len(self.data) == 0:
            raise ValueError(f"No sequences loaded from {data_folder}")
        # normalize
        original_shape = np.array(self.data).shape
        flat = np.array(self.data).reshape(-1, original_shape[-1])
        flat = self.scaler.fit_transform(flat) if scaler is None else self.scaler.transform(flat)
        self.data = flat.reshape(original_shape)
        self.labels = np.array(self.labels)

    def _load_data(self, data_folder):
        kinematics_folder = os.path.join(data_folder, 'kinematics')
        transcriptions_folder = os.path.join(data_folder, 'transcriptions')
        if not os.path.isdir(kinematics_folder):
            return
        kinematics_files = [f for f in os.listdir(kinematics_folder) if f.endswith('.csv')]
        all_sequences = []
        all_labels = []
        for filename in kinematics_files:
            kinematics_path = os.path.join(kinematics_folder, filename)
            transcription_path = os.path.join(transcriptions_folder, filename)
            try:
                kinematics_df = pd.read_csv(kinematics_path)
            except Exception as e:
                print(f"Could not read {kinematics_path}: {e}")
                continue
            if kinematics_df.shape[1] != self.expected_cols:
                print(f"Skipping {filename}: expected {self.expected_cols} columns, got {kinematics_df.shape[1]}")
                continue
            kinematics_data = kinematics_df.values
            labels = np.zeros(len(kinematics_data), dtype=int)
            if os.path.exists(transcription_path):
                try:
                    trans_df = pd.read_csv(transcription_path, header=None)
                    for _, row in trans_df.iterrows():
                        start_idx, end_idx, gesture_label = int(row[0]), int(row[1]), row[2]
                        if isinstance(gesture_label, str) and gesture_label.startswith('G'):
                            gnum = int(gesture_label[1:])
                            labels[start_idx:end_idx+1] = gnum
                except Exception as e:
                    print(f"Error parsing transcription {transcription_path}: {e}")
            seqs, seq_labels = self._create_sequences(kinematics_data, labels)
            all_sequences.extend(seqs)
            all_labels.extend(seq_labels)
        self.data = all_sequences
        self.labels = all_labels

    def _create_sequences(self, kinematics_data, labels):
        sequences = []
        seq_labels = []
        step_size = max(1, int(self.sequence_length * (1 - self.overlap)))
        for i in range(0, len(kinematics_data) - self.sequence_length + 1, step_size):
            sequences.append(kinematics_data[i:i + self.sequence_length])
            seq_labels.append(labels[i:i + self.sequence_length])
        return sequences, seq_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor(self.labels[idx])


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
    def __init__(self, input_size=76, hidden_size=128, num_layers=2, num_classes=16, dropout=0.2):
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
    def __init__(self, input_size=76, hidden_size=128, num_layers=2, num_classes=16, dropout=0.2):
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
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
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
    def __init__(self, input_size=76, hidden_channels=128, num_layers=4, kernel_size=3, 
                 num_classes=16, dropout=0.2):
        super(TCNModelWithAttention, self).__init__()
        
        layers = []
        in_ch = input_size
        for i in range(num_layers):
            layers.append(TCNBlock(in_ch, hidden_channels, kernel_size, dilation=2**i, dropout=dropout))
            in_ch = hidden_channels
        
        self.network = nn.Sequential(*layers)
        self.attention = ChangePointAttention(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_channels * 2, num_classes)
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x_transposed = x.transpose(1, 2)  # (batch, input_size, seq_len)
        out = self.network(x_transposed)  # (batch, hidden_channels, seq_len)
        out = out.transpose(1, 2)  # (batch, seq_len, hidden_channels)
        
        # Apply change point attention
        context, attn_weights, change_scores = self.attention(out)
        
        # Combine attention context with each timestep
        context_expanded = context.unsqueeze(1).expand(-1, out.size(1), -1)
        combined = torch.cat([out, context_expanded], dim=-1)
        
        combined = self.dropout(combined)
        logits = self.fc(combined)
        
        return logits


def train_model(model, train_loader, val_loader, num_epochs=25, lr=1e-3, weight_decay=1e-4):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_state = None
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            out_flat = outputs.reshape(-1, outputs.size(-1))
            tgt = y.reshape(-1)
            loss = criterion(out_flat, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            batches += 1
        avg_train = train_loss / max(1, batches)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                out_flat = outputs.reshape(-1, outputs.size(-1))
                tgt = y.reshape(-1)
                loss = criterion(out_flat, tgt)
                val_loss += loss.item()
                val_batches += 1
                _, pred = torch.max(out_flat, 1)
                total += tgt.size(0)
                correct += (pred == tgt).sum().item()
        avg_val = val_loss / max(1, val_batches)
        val_acc = 100.0 * correct / max(1, total)

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['val_acc'].append(val_acc)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        scheduler.step(avg_val)
        print(f"Epoch {epoch+1}/{num_epochs}: train={avg_train:.4f}, val={avg_val:.4f}, val_acc={val_acc:.2f}%")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate_model(model, test_loader, model_name, out_dir):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            out_flat = outputs.reshape(-1, outputs.size(-1))
            tgt = y.reshape(-1)
            _, pred = torch.max(out_flat, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(tgt.cpu().numpy())

    all_preds = np.array(all_preds, dtype=int)
    all_targets = np.array(all_targets, dtype=int)

    if all_targets.size == 0:
        return None 

    present_labels = np.unique(np.concatenate([all_targets, all_preds]))
    present_labels_sorted = np.sort(present_labels).tolist()

    target_names = []
    for lab in present_labels_sorted:
        if lab == 0:
            target_names.append('Background')
        else:
            target_names.append(f'G{lab}')

    # Get classification report as dict
    report = classification_report(all_targets, all_preds, labels=present_labels_sorted,
                                   target_names=target_names, zero_division=0, output_dict=True)
    
    # Get classification report as string for saving
    report_str = classification_report(all_targets, all_preds, labels=present_labels_sorted,
                                       target_names=target_names, zero_division=0)

    cm = confusion_matrix(all_targets, all_preds, labels=present_labels_sorted)

    # Save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    fname = os.path.join(out_dir, f'{model_name}_confusion.png')
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()

    # Calculate standard metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, labels=present_labels_sorted, average='weighted', zero_division=0
    )

    # Beta distribution for confidence interval
    successes = int(np.sum(all_preds == all_targets))
    trials = len(all_targets)
    alpha, beta_param = successes + 1, (trials - successes) + 1
    dist = sp_beta(alpha, beta_param)
    ci_low, ci_high = dist.interval(0.95)

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'classification_report_dict': report,
        'classification_report_str': report_str,
        'present_labels': present_labels_sorted,
        'confusion_matrix': cm,
        'beta_CI_95': (float(ci_low), float(ci_high))
    }


def fit_beta_from_samples(samples):
    """
    Fit Beta distribution via method of moments to samples in (0,1).
    Return dict with alpha, beta, mean, std, ci95 (2-tuple).
    If variance is zero or fit fails, return None.
    """
    samples = np.array(samples, dtype=float)
    samples = samples[(samples > 0.0) & (samples < 1.0)]
    if len(samples) == 0:
        return None
    m = np.mean(samples)
    v = np.var(samples, ddof=1) if len(samples) > 1 else 0.0
    if v <= 0 or m <= 0 or m >= 1:
        mean = np.mean(samples) if len(samples) > 0 else 0.0
        std = np.std(samples, ddof=1) if len(samples) > 1 else 0.0
        return {
            'alpha': None, 'beta': None,
            'mean': mean, 'std': std,
            'ci95': (max(0.0, mean - 1.96 * std), min(1.0, mean + 1.96 * std))
        }

    common = (m * (1 - m) / v) - 1.0
    if common <= 0:
        return None
    a = m * common
    b = (1 - m) * common
    try:
        dist = sp_beta(a, b)
        mean = dist.mean()
        std = dist.std()
        ci_low, ci_high = dist.ppf(0.025), dist.ppf(0.975)
        return {'alpha': float(a), 'beta': float(b), 'mean': float(mean), 'std': float(std), 'ci95': (float(ci_low), float(ci_high))}
    except Exception:
        return None


def run_full_experiment(root_base, output_base, sequence_length=128, overlap=0.5,
                        batch_size=32, num_epochs=30, lr=1e-3, expected_cols=76):
    """
    root_base: top folder containing activity folders (Needle_Passing, Knot_Tying, Suturing)
    structure expected:
    <root_base>/<activity>/<User X>/Fold Y/{train,test}/kinematics/*.csv and transcriptions/*.csv
    
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

        model_fold_metrics = defaultdict(list) 
        pooled_predictions = defaultdict(lambda: {'all_preds': [], 'all_targets': []})

        users = [u for u in activity_path.iterdir() if u.is_dir()]
        if not users:
            print(f"No user folders under {activity_path}, skipping")
            continue

        for user in users:
            folds = [f for f in user.iterdir() if f.is_dir()]
            if not folds:
                continue
            for fold in folds:
                fold_name = f"{user.name}_{fold.name}"
                print(f"\n--- Processing fold: {fold_name} ---")
                fold_out = activity_out / fold_name
                fold_out.mkdir(exist_ok=True, parents=True)

                train_dir = fold / 'train'
                test_dir = fold / 'test'

                if not train_dir.exists() or not test_dir.exists():
                    print(f"Missing train/test in {fold}. Skipping fold.")
                    continue

                try:
                    train_ds = GestureDataset(str(train_dir), sequence_length=sequence_length,
                                              overlap=overlap, expected_cols=expected_cols, scaler=None)
                except Exception as e:
                    print(f"Could not create train dataset for {fold_name}: {e}")
                    continue

                scaler = train_ds.scaler
                try:
                    test_ds = GestureDataset(str(test_dir), sequence_length=sequence_length,
                                             overlap=overlap, expected_cols=expected_cols, scaler=scaler)
                except Exception as e:
                    print(f"Could not create test dataset for {fold_name}: {e}")
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
                    print(f"Training {model_name} on {fold_name}")
                    try:
                        trained_model, history = train_model(model, train_loader, val_loader,
                                                             num_epochs=num_epochs, lr=lr)
                    except Exception as e:
                        print(f"Training failed for {model_name} on {fold_name}: {e}")
                        continue

                    model_path = fold_out / f"{model_name.lower()}_best.pth"
                    torch.save(trained_model.state_dict(), str(model_path))

                    metrics = evaluate_model(trained_model, test_loader, model_name, str(fold_out))
                    if metrics is None:
                        print(f"No test data for {fold_name}/{model_name}, skipping.")
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

                    model_fold_metrics[model_name].append({
                        'user': user.name,
                        'fold': fold.name,
                        'fold_name': fold_name,
                        'accuracy': metrics['accuracy'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1_score': metrics['f1_score'],
                        'beta_CI_95': metrics['beta_CI_95'],
                        'present_labels': metrics['present_labels'],
                        'classification_report': metrics['classification_report_str']
                    })

        # Generate activity-level summary with pooled metrics
        activity_summary = {}
        activity_report_path = activity_out / 'activity_classification_reports.txt'
        
        with open(activity_report_path, 'w') as report_file:
            report_file.write("="*80 + "\n")
            report_file.write(f"ACTIVITY: {activity.upper()} (WITH CHANGE POINT ATTENTION)\n")
            report_file.write("="*80 + "\n\n")
            
            for model_name, folds_list in model_fold_metrics.items():
                if len(folds_list) == 0:
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
                    
                    report_file.write(f"Overall Metrics (Pooled across all folds):\n")
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
                    
                    activity_summary[model_name] = {
                        'num_folds': len(folds_list),
                        'pooled_accuracy': float(pooled_accuracy),
                        'pooled_precision': float(pooled_precision),
                        'pooled_recall': float(pooled_recall),
                        'pooled_f1': float(pooled_f1),
                        'beta_CI_95': (float(ci_low), float(ci_high)),
                        'folds_detail': folds_list
                    }
                
                # Per-fold results
                df_rows = []
                for f in folds_list:
                    df_rows.append({
                        'user': f['user'],
                        'fold': f['fold'],
                        'fold_name': f['fold_name'],
                        'accuracy': f['accuracy'],
                        'precision': f['precision'],
                        'recall': f['recall'],
                        'f1_score': f['f1_score'],
                        'ci_low': f['beta_CI_95'][0],
                        'ci_high': f['beta_CI_95'][1],
                        'present_labels': ",".join(map(str, f['present_labels']))
                    })
                pd.DataFrame(df_rows).to_csv(activity_out / f"{model_name}_per_fold_results.csv", index=False)

        overall_summary[activity] = activity_summary

        with open(activity_out / 'activity_summary.json', 'w') as fh:
            json.dump(activity_summary, fh, indent=2)

    # Overall summary across activities
    with open(output_base / 'overall_summary.json', 'w') as fh:
        json.dump(overall_summary, fh, indent=2)
    
    # Create overall ranking report
    with open(output_base / 'overall_ranking.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("OVERALL PERFORMANCE RANKING ACROSS ALL ACTIVITIES (WITH ATTENTION)\n")
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

    print("\nLOSO Experiment with Change Point Attention finished. Results written to:", output_base)
    return overall_summary


if __name__ == "__main__":
    ROOT = r"D:\AlecCotton\Combined Segmentation and Classification\LOSO"
    OUT = r"D:\AlecCotton\dl_seg_clf_loso_attention"

    SEQ_LEN = 128
    OVERLAP = 0.5
    BATCH = 32
    EPOCHS = 30
    LR = 1e-3
    EXPECTED_COLS = 76

    run_full_experiment(ROOT, OUT, sequence_length=SEQ_LEN, overlap=OVERLAP,
                        batch_size=BATCH, num_epochs=EPOCHS, lr=LR, expected_cols=EXPECTED_COLS)
