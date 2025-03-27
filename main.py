import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from opacus import PrivacyEngine
import matplotlib.pyplot as plt
import pandas as pd
import logging
import os

# ------------------------------
# 設定 logging 輸出到檔案及 console
# ------------------------------
log_filename = "training_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# ==========================================
# 1. 設定超參數
# ==========================================
BATCH_SIZE = 256
EPOCHS = 20
LR = 0.001  # 降低學習率
# EPSILON 值改為包含更低的值，噪聲量較大時，target_epsilon 較低
EPSILONS = [0.01, 0.5, 5.0, 20.0]
MAX_GRAD_NORM = 1.0  # clipping 的最大梯度範圍

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 載入資料集 (MNIST)
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 3. 定義簡單的 CNN 模型
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 輸出 [16, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # 輸出 [32, 7, 7]
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==========================================
# 4. 訓練與測試函數
# ==========================================
def train_model(model, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100.0 * correct / total
    msg = f"Epoch [{epoch + 1}], Loss: {loss.item():.4f}, Train Accuracy: {train_acc:.2f}%"
    print(msg)
    logging.info(msg)

def img_test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100.0 * correct / total
    return acc

# ==========================================
# 主程式入口
# ==========================================
if __name__ == "__main__":
    # -------------------------------
    # (A) Baseline (無差分隱私) 訓練
    # -------------------------------
    logging.info("==== Baseline (No DP) Training ====")
    print("==== Baseline (No DP) Training ====")
    model_baseline = SimpleCNN().to(DEVICE)
    optimizer = optim.SGD(model_baseline.parameters(), lr=LR)

    baseline_test_acc_list = []
    for epoch in range(EPOCHS):
        train_model(model_baseline, train_loader, optimizer, epoch)
        test_acc = img_test_model(model_baseline, test_loader)
        msg = f"Test Accuracy: {test_acc:.2f}%"
        print(msg + "\n")
        logging.info(msg)
        baseline_test_acc_list.append(test_acc)
    logging.info("Baseline training finished.\n")
    print("Baseline training finished.\n")

    # -------------------------------
    # (B) 差分隱私訓練 (DPSGD)
    # -------------------------------
    dp_results = []
    for eps in EPSILONS:
        msg = f"==== DP Training with ε = {eps} ===="
        logging.info(msg)
        print(msg)
        model_dp = SimpleCNN().to(DEVICE)
        optimizer_dp = optim.SGD(model_dp.parameters(), lr=LR, momentum=0)

        # 利用新版 Opacus 的 make_private_with_epsilon 方法進行隱私保護
        privacy_engine = PrivacyEngine()
        model_dp, optimizer_dp, train_loader_dp = privacy_engine.make_private_with_epsilon(
            module=model_dp,
            optimizer=optimizer_dp,
            data_loader=train_loader,
            target_epsilon=eps,
            target_delta=1e-5,
            epochs=EPOCHS,
            max_grad_norm=MAX_GRAD_NORM
        )

        dp_test_acc_list = []
        for epoch in range(EPOCHS):
            model_dp.train()
            for images, labels in train_loader_dp:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer_dp.zero_grad()
                outputs = model_dp(images)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer_dp.step()

            test_acc = img_test_model(model_dp, test_loader)
            dp_test_acc_list.append(test_acc)
            msg = f"Epoch [{epoch + 1}/{EPOCHS}], ε={eps}, Test Accuracy: {test_acc:.2f}%"
            print(msg)
            logging.info(msg)
        dp_results.append((eps, dp_test_acc_list))
        print("\n")
        logging.info("\n")
    logging.info("DP training finished.\n")
    print("DP training finished.\n")

    # -------------------------------
    # 結果展示
    # -------------------------------
    # (A) 表格呈現不同 epsilon 在最後一個 epoch 的準確率
    final_epsilons = []
    final_accuracies = []
    for eps, acc_list in dp_results:
        final_epsilons.append(eps)
        final_accuracies.append(acc_list[-1])

    df_results = pd.DataFrame({
        'Epsilon': final_epsilons,
        f'Accuracy@Epoch{EPOCHS}': final_accuracies
    })
    msg = "=== Differential Privacy Results ===\n" + df_results.to_string()
    print(msg)
    logging.info(msg)

    # (B) Accuracy 曲線圖 (DP 部分)
    plt.figure(figsize=(8, 6))
    for eps, acc_list in dp_results:
        plt.plot(range(1, EPOCHS + 1), acc_list, marker='o', label=f'ε = {eps}')
    plt.title('Test Accuracy vs. Epochs (DP Training)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    dp_plot_filename = "dp_training_accuracy.png"
    plt.savefig(dp_plot_filename)
    logging.info(f"DP training accuracy plot saved to {dp_plot_filename}")
    plt.show()

    # 額外：Baseline 與 DP (各 epsilon) 最終 accuracy 的比較曲線圖
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, EPOCHS + 1), baseline_test_acc_list, marker='o', label='Baseline (No DP)')
    for eps, acc_list in dp_results:
        plt.plot(range(1, EPOCHS + 1), acc_list, marker='o', label=f'DP (ε={eps})')
    plt.title('Test Accuracy Comparison: Baseline vs. DP')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    comparison_plot_filename = "baseline_vs_dp_accuracy.png"
    plt.savefig(comparison_plot_filename)
    logging.info(f"Baseline vs. DP accuracy comparison plot saved to {comparison_plot_filename}")
    plt.show()
