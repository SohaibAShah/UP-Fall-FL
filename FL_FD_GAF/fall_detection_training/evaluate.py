import torch
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

def evaluate_model(model, test_data, classes):
    """Evaluate the model on the test dataset and visualize results.
    
    Args:
        model (nn.Module): Trained model.
        test_data (dict): Test dataset.
        classes (list): List of class names.
    """
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)
    y_test, y_predict = [], []
    
    for batch_id, batch in enumerate(test_loader):
        data = batch[0]
        target = torch.squeeze(batch[1]).int()
        target = torch.tensor(target, dtype=torch.int64)

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        output = model(data)
        y_test.extend(target.cpu().numpy())
        y_predict.extend(torch.argmax(output, dim=1).cpu().numpy())
    
    # Classification report
    print(classification_report(y_test, y_predict, target_names=classes, digits=4))
    
    # Confusion matrix
    plt.figure(dpi=150, figsize=(6, 4))
    mat = confusion_matrix(y_test, y_predict)
    df_cm = pd.DataFrame(mat, index=classes, columns=classes)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.show()