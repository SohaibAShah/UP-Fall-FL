import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from src.dataset.Dataloader import FallDetectionDataset
from src.models.simplemlp import SimpleMLP # Import the new MLP model

def train_epoch(model, dataloader, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X)
    return total_loss / len(dataloader.dataset)

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Evaluates the model and returns loss and metrics."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_y, all_preds = [], []
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(X)
        all_preds.append(logits.argmax(1).cpu())
        all_y.append(y.cpu())
        
    y_true = torch.cat(all_y).numpy()
    y_pred = torch.cat(all_preds).numpy()
    
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    acc = float((y_true == y_pred).mean())
    avg_loss = total_loss / len(dataloader.dataset)
    
    return avg_loss, acc, p, r, f1

def main(args):
    """Main function to run the centralized training."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # --- Updated Data Loading ---
    # Point to the directory with the cleaned CSV files
    data_dir = Path("data/upfall/processed/sensor")
    
    if not any(data_dir.iterdir()):
        print(f"No cleaned CSV files found in {data_dir}. Run the preparation script first.")
        return
        
    # Instantiate the dataset for each split (windowing arguments removed)
    ds_train = FallDetectionDataset(data_dir, split='train')
    ds_val = FallDetectionDataset(data_dir, split='val')
    ds_test = FallDetectionDataset(data_dir, split='test')

    # Create DataLoaders
    dl_train = DataLoader(ds_train, batch_size=args.bs, shuffle=True, num_workers=2, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=args.bs, shuffle=False, num_workers=2, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=args.bs, shuffle=False, num_workers=2, pin_memory=True)

    # --- Use the MLP Model ---
    # The input_size is 6, as we are using 6 sensor axes and skipping the Time column.
    model = SimpleMLP(input_size=6, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_f1, best_model_state = 0.0, None

    print("--- Starting Training with MLP ---")
    for ep in range(1, args.epochs + 1):
        train_loss = train_epoch(model, dl_train, optimizer, device)
        val_loss, val_acc, val_p, val_r, val_f1 = evaluate_model(model, dl_val, device)
        
        print(f"Epoch {ep:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}, F1: {val_f1:.3f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  -> New best model saved with F1: {best_f1:.3f}")

    if best_model_state is None:
        print("Training finished, but no best model was saved.")
        return
        
    # Load the best model and evaluate on the test set
    model.load_state_dict(best_model_state)
    test_loss, test_acc, test_p, test_r, test_f1 = evaluate_model(model, dl_test, device)
    print("\n--- Final Test Set Evaluation ---")
    print(f"Test Accuracy: {test_acc:.3f}")
    print(f"Test Precision: {test_p:.3f} | Recall: {test_r:.3f} | F1-Score: {test_f1:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced training script for fall detection.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--bs", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    # Window and stride arguments are removed
    args = parser.parse_args()
    main(args)
