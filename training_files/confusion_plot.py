import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("./resnet18_gender/confusion_matrix.csv", index_col=0)  # index_col=0 = true_male/true_female
cm = df.values  # 2x2 numpy array

classes_true = df.index.to_list()
classes_pred = df.columns.to_list()

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)


fig, ax = plt.subplots(figsize=(10, 10))

im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.set_ylabel("Proportion", rotation=-90, va="bottom")

classes_true = df.index.to_list()
classes_pred = df.columns.to_list()

ax.set_xticks(np.arange(len(classes_pred)))
ax.set_yticks(np.arange(len(classes_true)))
ax.set_xticklabels(classes_pred, rotation=45, ha="right")
ax.set_yticklabels(classes_true)

ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Gender Confusion Matrix")

threshold = cm_norm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = int(cm[i, j])
        pct = cm_norm[i, j] * 100.0
        txt = f"{count}\n{pct:.1f}%"
        color = "white" if cm_norm[i, j] > threshold else "black"
        ax.text(
            j, i, txt,
            ha="center", va="center",
            color=color,
            fontsize=10, fontweight="bold"
        )

plt.tight_layout()
plt.savefig("./resnet18_gender/confusion_gender_blue.png", dpi=200)
plt.show()