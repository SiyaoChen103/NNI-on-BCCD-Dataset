loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(opt_model.parameters())

num_epochs = 50
for epoch in range(num_epochs):
    opt_model.train()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = opt_model(inputs)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Loss at epoch {epoch+1}: {loss.item():.4f}')
