import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_extraction import extract_features, extract_features_with_augmentation
from model.model_pytorch import get_model
import config

class EmotionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_data(dataset_path, use_augmentation=True, fixed_length=200):
    """Load RAVDESS dataset with optional augmentation."""
    features = []
    labels = []
    
    emotion_map = {
        '01': 6, # Neutral
        '02': 6, # Calm -> Neutral
        '03': 0, # Happy
        '04': 1, # Sad
        '05': 2, # Angry
        '06': 3, # Fear
        '07': 5, # Disgust
        '08': 4  # Surprise
    }
    
    # Map index back to string for config/display if needed
    # config.EMOTIONS = ['Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Disgust', 'Neutral']
    
    audio_files = glob.glob(os.path.join(dataset_path, '**', '*.wav'), recursive=True)
    
    if not audio_files:
        print(f"⚠️ No files found in {dataset_path}")
        return None, None
    
    print(f"📂 Found {len(audio_files)} audio files. Extracting features...")
    
    for i, filepath in enumerate(audio_files):
        try:
            filename = os.path.basename(filepath)
            parts = filename.split('-')
            
            if len(parts) >= 3:
                emotion_code = parts[2]
                label = emotion_map.get(emotion_code, None)
                
                if label is not None:
                    if use_augmentation:
                        # Extract multiple features (original + augmented)
                        aug_features = extract_features_with_augmentation(filepath)
                        for feat in aug_features:
                            # Pad/Truncate each
                            if feat.shape[1] < fixed_length:
                                feat = np.pad(feat, ((0,0), (0, fixed_length - feat.shape[1])), mode='constant')
                            else:
                                feat = feat[:, :fixed_length]
                            features.append(feat)
                            labels.append(label)
                    else:
                        feat = extract_features(filepath, fixed_length=fixed_length)
                        if feat is not None:
                            features.append(feat)
                            labels.append(label)
                            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(audio_files)} files...")
        except Exception as e:
            continue
            
    return np.array(features), np.array(labels)

def train_pytorch_model(dataset_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on {device}")
    
    X, y = load_data(dataset_path)
    if X is None or len(X) == 0:
        print("❌ No data found. Training aborted.")
        return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_dataset = EmotionDataset(X_train, y_train)
    test_dataset = EmotionDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Calculate class weights for imbalance
    class_counts = np.bincount(y)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(config.EMOTIONS)
    class_weights = torch.FloatTensor(weights).to(device)
    
    model = get_model(num_classes=len(config.EMOTIONS)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50
    best_acc = 0
    
    print("\n🎬 Starting Training Loop")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        val_acc = accuracy_score(all_targets, all_preds)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "emotion_model.pth"))
            print(f"🌟 Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f} - Val Acc: {val_acc:.4f} (Saved!)")
        elif (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f} - Val Acc: {val_acc:.4f}")

    print(f"\n✅ Training Finished. Best Val Accuracy: {best_acc:.4f}")
    
    # Final Evaluation Report
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "emotion_model.pth")))
    model.eval()
    final_preds = []
    final_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            final_preds.extend(preds.cpu().numpy())
            final_targets.extend(targets.cpu().numpy())
            
    print("\n📊 Classification Report:")
    print(classification_report(final_targets, final_preds, target_names=config.EMOTIONS))
    
    # Save Confusion Matrix plot
    cm = confusion_matrix(final_targets, final_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=config.EMOTIONS, yticklabels=config.EMOTIONS)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(os.path.dirname(__file__), "confusion_matrix.png"))
    print(f"🖼️ Confusion matrix saved to model/confusion_matrix.png")

if __name__ == '__main__':
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'dataset', 'RAVDESS')
    train_pytorch_model(dataset_path)
