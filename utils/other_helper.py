
# some functions might be useful
import numpy as np
import torch
import random

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total:', total_num, 'Trainable:', trainable_num)
    return {'Total': total_num, 'Trainable': trainable_num}


def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

def viz_plot(viz, nll_loss, nll_loss_eval, acc, epoch):
    if epoch == 0:
        viz.line(X=np.array([epoch]), Y=np.array([nll_loss]), win="loss", name="train_loss")
    else:
        viz.line(X=np.array([epoch]), Y=np.array([nll_loss]), win="loss", name="train_loss", update="append")

    viz.line(X=np.array([epoch]), Y=np.array([nll_loss_eval]), win="loss", name="val_loss", update="append")
    viz.line(X=np.array([epoch]), Y=np.array([acc]), win="loss", name="val_auc", update="append")


def compute_loss(model, optimizer, images, target, loss_fn, train_mode):

    predicted = model(images)
    loss = loss_fn(predicted, target)

    if train_mode:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    else:
        return loss.item(), torch.argmax(predicted, dim=1).float()

    return loss.item()