import config
import torch
import random
import torch.nn as nn
from tqdm import tqdm

def cse_loss(student_out, teacher_out):
    student_out = torch.softmax(student_out, dim=-1)
    teacher_out = torch.softmax(teacher_out, dim=-1)
    return -(teacher_out*torch.log(student_out + 1e-8)).sum(dim=1).mean()


def t_n_s_loss_fn(
    t_out_1, 
    t_out_2, 
    t_out_4, 
    s_out_1, 
    s_out_2, 
    s_out_4, 
    labels, 
):
    cce_l = nn.CrossEntropyLoss()(s_out_4, labels.view(-1))
    s_out_4 = nn.LogSoftmax()(s_out_4/config.temperature)
    t_out_4 = torch.softmax(t_out_4/config.temperature, dim=-1)
    kde_loss = nn.KLDivLoss()(s_out_4, t_out_4)
    
    return kde_loss*(0.7*config.temperature*config.temperature) + cce_l*0.3


def loss_fn(prediction, ground_truth):
    return nn.CrossEntropyLoss()(prediction, ground_truth)


def accuracy_fn(out, ground_truth):
    prediction = torch.argmax(out, dim=-1)
    return ((prediction.reshape(-1, 1) == ground_truth)*1.0).mean()
    

def train_fn(
    model, 
    dataloader, 
    optimizer, 
    scheduler, 
    device, 
    t_n_s=False, 
    include_t_n_s_mid_loss=False,
    teachers_input_student_ratio=10,
):
    running_loss = 0
    running_acc = 0
    model.train()
    for num, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        for p in model.parameters():
            p.grad = None
        imgs = data["image"].to(device)
        labels = data["label"].to(device)
        if t_n_s :
            (t_out_1, t_out_2, _, t_out_4), (s_out_1, s_out_2, _, s_out_4) = model(imgs, teachers_input_student_ratio)
            loss = t_n_s_loss_fn(
                t_out_1, 
                t_out_2, 
                t_out_4, 
                s_out_1, 
                s_out_2, 
                s_out_4, 
                labels
            )
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


def eval_fn(model, dataloader, device, t_n_s=False, include_t_n_s_mid_loss=False):
    running_loss = 0
    running_acc = 0
    model.eval()
    for num, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        imgs = data["image"].to(device)
        labels = data["label"].to(device)
        with torch.no_grad():
            if t_n_s:
                (t_out_1, t_out_2, _, t_out_4), (s_out_1, s_out_2, _, s_out_4) = model(imgs)
                loss = t_n_s_loss_fn(
                    t_out_1, 
                    t_out_2, 
                    t_out_4,
                    s_out_1, 
                    s_out_2, 
                    s_out_4, 
                    labels.view(-1),
                )
            else:
                _, _, _, s_out_4= model(imgs)
                loss = loss_fn(s_out_4, labels.view(-1))
        running_loss += loss.item()
        acc = accuracy_fn(s_out_4, labels)
        running_acc += acc
    
    return running_acc/len(dataloader), running_loss/len(dataloader)