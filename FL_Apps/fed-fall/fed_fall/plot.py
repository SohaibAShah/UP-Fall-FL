import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("client_0_eval_metrics.csv")  # Change to your client ID

plt.figure()
plt.plot(df["round"], df["eval_loss"], label="Eval Loss")
plt.plot(df["round"], df["eval_acc"], label="Eval Accuracy")
plt.plot(df["round"], df["F1-score"], label="F1-score")
plt.plot(df["round"], df["Precision"], label="Precision")
plt.plot(df["round"], df["Recall"], label="Recall")
plt.xlabel("Round")
plt.legend()
plt.title("Client 0 Evaluation Metrics")
plt.show()