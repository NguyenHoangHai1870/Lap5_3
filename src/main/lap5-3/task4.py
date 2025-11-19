# ===== Task 4: Huấn luyện mô hình =====
import torch.optim as optim

# Chọn device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function & optimizer
criterion = nn.CrossEntropyLoss(ignore_index=PAD_TAG)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        outputs = outputs.view(-1, num_tags)
        y_batch_flat = y_batch.view(-1)

        loss = criterion(outputs, y_batch_flat)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")
