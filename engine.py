import torch
import random
import torch.nn as nn
from tqdm import tqdm

def cse_loss(student_out, teacher_out):     
    return -(teacher_out*torch.log(student_out)).sum(dim=1).mean()

def t_n_s_loss_fn(t_out_1, t_out_2, t_out_4, s_out_1, s_out_2, s_out_4, labels):
    mse_l_1 = nn.MSELoss()(t_out_1, s_out_1)
    mse_l_2 = nn.MSELoss()(t_out_2, s_out_2)
    if random.random() > 0.5:
        cce_l = nn.CrossEntropyLoss()(s_out_4, labels.view(-1))
    else:
        # cce_l = cse_loss(s_out_4, t_out_4)

        cce_l = nn.KLDivLoss()(s_out_4, t_out_4)
    
    return (mse_l_1 + mse_l_2 + cce_l)/3

def loss_fn(prediction, ground_truth):
    return nn.CrossEntropyLoss()(prediction, ground_truth)

def accuracy_fn(out, ground_truth):
    prediction = torch.argmax(torch.softmax(out, dim=-1), dim=-1)
    return ((prediction == ground_truth)*1.0).sum()/ground_truth.shape[0]
    

def train_fn(model, dataloader, optimizer, scheduler, device, t_n_s=False):
    running_loss = 0
    running_acc = 0
    model.train()
    for num, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        for p in model.parameters():
            p.grad = None
        imgs = data["image"].to(device)
        labels = data["label"].to(device)
        if t_n_s :
            (t_out_1, t_out_2, _, t_out_4), (s_out_1, s_out_2, _, s_out_4) = model(imgs)
            loss = t_n_s_loss_fn(t_out_1, t_out_2, t_out_4, s_out_1, s_out_2, s_out_4, labels)
        else:
            _, _, _, s_out_4= model(imgs)
            loss = loss_fn(s_out_4, labels.view(-1))
        running_loss += loss.item()
        acc = accuracy_fn(s_out_4, labels)
        running_acc += acc
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    return running_acc/len(dataloader), running_loss/len(dataloader)

def eval_fn(model, dataloader, device, t_n_s=False):
    running_loss = 0
    running_acc = 0
    model.eval()
    for num, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        imgs = data["image"].to(device)
        labels = data["label"].to(device)
        with torch.no_grad():
            if t_n_s:
                (t_out_1, t_out_2, _, t_out_4), (s_out_1, s_out_2, _, s_out_4) = model(imgs)
                loss = t_n_s_loss_fn(t_out_1, t_out_2, t_out_4, s_out_1, s_out_2, s_out_4, labels.view(-1))
            else:
                _, _, _, s_out_4= model(imgs)
                loss = loss_fn(s_out_4, labels.view(-1))
        running_loss += loss.item()
        acc = accuracy_fn(s_out_4, labels)
        running_acc += acc
    
    return running_acc/len(dataloader), running_loss/len(dataloader)

