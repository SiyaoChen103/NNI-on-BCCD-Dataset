import torch

from nni.retiarii.oneshot.pytorch import DartsTrainer

def accuracy(output, target):
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    return {"acc1": (predicted == target).sum().item() / batch_size}

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

trainer = DartsTrainer(
    model=model,
    loss=criterion,
    metrics=lambda output, target: accuracy(output, target),
    optimizer=optimizer,
    num_epochs=10,
    dataset=train_dataset,
    batch_size=64,
    log_frequency=10
    )
trainer.fit()
