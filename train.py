import torch
import numpy as np
import copy
import time


def train_model(model,
                dataloaders,
                criterion,
                optimizer,
                scheduler,
                filename,
                device,
                num_epochs=25):
    since = time.time()

    best_acc = 0
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]["lr"]]
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            time_eplased = time.time() - since
            print("Time elapsed {:.0f}m {:.0f}s".format(time_eplased // 60, time_eplased % 60))
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                state = {
                    "state_dict": model.state_dict(),
                    "best_acc": best_acc,
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase == "valid":
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step(epoch_loss)
            if phase == "train":
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        print("Optimizer learning rate: {:.7f}".format(optimizer.param_groups[0]["lr"]))
        LRs.append(optimizer.param_groups[0]["lr"])
        print()

    time_eplased = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_eplased // 60, time_eplased % 60))
    print("Best val Acc: {:4f}".format(best_acc))

    model.load_state_dict(best_model_weights)

    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs