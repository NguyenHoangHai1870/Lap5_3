# ===== Task 5: Đánh giá mô hình =====
def evaluate(model, data_loader):
    model.eval()
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=-1)

            mask = y_batch != PAD_TAG
            total_correct += (preds == y_batch).masked_select(mask).sum().item()
            total_tokens += mask.sum().item()
    return total_correct / total_tokens

# Accuracy train/dev
train_acc = evaluate(model, train_loader)
dev_acc = evaluate(model, dev_loader)
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Dev Accuracy  : {dev_acc:.4f}")
