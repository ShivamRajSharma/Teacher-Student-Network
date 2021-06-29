import config
import engine
import TeacherStudentNetwork
import dataloader

import transformers
import torch
import torch.nn as nn
import pickle
from tensorflow.keras.datasets import cifar100
import albumentations as alb
# import torchvision
from sklearn.model_selection import train_test_split

def run():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    train_transforms = alb.Compose([
        alb.Normalize(config.cifar100_mean, config.cifar100_std),
        alb.Flip(p=0.2),
        alb.Transpose(p=0.2)
    ])

    val_transforms = alb.Compose([
        alb.Normalize(config.cifar100_mean, config.cifar100_std)
    ])

    train_data = dataloader.DataLoader(x_train, y_train, train_transforms)

    val_data = dataloader.DataLoader(x_test, y_test, val_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config.Batch_Size,
        num_workers=4,
        shuffle=True,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=config.Batch_Size,
        num_workers=4,
        pin_memory=True
    )

    teacher_model = TeacherStudentNetwork.Teacher(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        num_conv=config.num_conv,
        num_classes=config.num_classes,
        dropout=config.dropout
    )

    student_model = TeacherStudentNetwork.Student(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        num_classes=config.num_classes,
        dropout=config.dropout
    )

    if torch.cuda.is_available():
        accelarator = 'cuda'
        device = torch.device(accelarator)
        torch.backends.cudnn.benchmark = True
    else:
        accelarator = 'cpu'
        device = torch.device(accelarator)

    
    teacher_model.to(device)
    student_model.to(device)

    teacher_optimizer = transformers.AdamW(teacher_model.parameters(), lr=config.lr)
    student_optimizer = transformers.AdamW(student_model.parameters(), lr=config.lr)

    num_training_steps = config.Epochs*len(train_data)//config.Batch_Size

    teacher_scheduler = transformers.get_linear_schedule_with_warmup(
        teacher_optimizer,
        num_warmup_steps=config.warmup_stes*num_training_steps,
        num_training_steps=num_training_steps
    )

    student_scheduler = transformers.get_linear_schedule_with_warmup(
        student_optimizer,
        num_warmup_steps=config.warmup_stes*num_training_steps,
        num_training_steps=num_training_steps
    )

    best_loss = 1e4
    best_acc = 0.0
    teachers_best_dict = teacher_model.state_dict

    teachers_train_acc = []
    teachers_train_loss = []
    teachers_val_acc = []
    teachers_val_loss = []

    students_train_acc = []
    students_train_loss = []
    students_val_acc = []
    students_val_loss = []

    # print('--------- [INFO] STARTING TEACHERS TRAINING ---------')
    # for epoch in range(config.Epochs):
    #     train_accuracy, train_loss = engine.train_fn(teacher_model, train_loader, teacher_optimizer, teacher_scheduler, device)
    #     val_accuracy, val_loss = engine.eval_fn(teacher_model, val_loader, device)
    #     teachers_train_acc.append(train_accuracy)
    #     teachers_train_loss.append(train_loss)
    #     teachers_val_acc.append(val_accuracy)
    #     teachers_val_loss.append(val_loss)

    #     print(f'EPOCH -> {epoch+1}/{config.Epochs} | TEACHERS TRAIN LOSS = {train_loss} | TEACHERS TRAIN ACC - {train_accuracy*100}% | TEACHERS VAL LOSS = {val_loss} | TEACHER VAL ACC = {val_accuracy*100}%')
    #     if best_acc < val_accuracy:
    #         best_acc = val_accuracy
    #         teachers_best_dict = teacher_model.state_dict()

    # torch.save(teachers_best_dict, config.teacher_model_path)

    # print('--------- [INFO] STARTING STUDENTS TRAINING ---------')
    # for epoch in range(config.Epochs):
    #     train_accuracy, train_loss = engine.train_fn(student_model, train_loader, student_optimizer, student_scheduler, device)
    #     val_accuracy, val_loss = engine.eval_fn(student_model, val_loader, device)
    #     students_train_acc.append(train_accuracy)
    #     students_train_loss.append(train_loss)
    #     students_val_acc.append(val_accuracy)
    #     students_val_loss.append(val_loss)

    #     print(f'EPOCH -> {epoch+1}/{config.Epochs} | STUDENTS TRAIN LOSS = {train_loss} | STUDENTS TRAIN ACC = {train_accuracy*100}% | STUDENTS VAL LOSS = {val_loss} | STUDENTS VAL ACC = {val_accuracy*100}%')
    
    teacher_student_model = TeacherStudentNetwork.TeacherStudentNetwork(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        teacher_num_conv=config.num_conv,
        teacher_weights_path=config.teacher_model_path,
        num_classes=config.num_classes,
        dropout=config.dropout
    )

    teacher_student_model.to(device)

    teacher_student_optimizer = transformers.AdamW(teacher_student_model.parameters(), lr=config.lr)


    teacher_student_scheduler = transformers.get_linear_schedule_with_warmup(
        teacher_student_optimizer,
        num_warmup_steps=config.warmup_stes*num_training_steps,
        num_training_steps=num_training_steps
    )

    teacher_student_best_dict = teacher_student_model.student.state_dict()

    teacher_student_train_acc = []
    teacher_student_train_loss = []
    teacher_student_val_acc = []
    teacher_student_val_loss = []

    print('--------- [INFO] STARTING TEACHER STUDENT TRAINING ---------')
    for epoch in range(config.Epochs):
        train_accuracy, train_loss = engine.train_fn(teacher_student_model, train_loader, teacher_student_optimizer, teacher_student_scheduler, device, t_n_s=True)
        val_accuracy, val_loss = engine.eval_fn(teacher_student_model, val_loader, device, t_n_s=True)
        teacher_student_train_acc.append(train_accuracy)
        teacher_student_train_loss.append(train_loss)
        teacher_student_val_acc.append(val_accuracy)
        teacher_student_val_loss.append(val_loss)

        print(f'EPOCH -> {epoch+1}/{config.Epochs} | TEACHER STUDENT TRAIN LOSS = {train_loss}  | TEACHER STUDENT TRAIN ACC = {train_accuracy*100}% | TEACHER STUDENT VAL LOSS = {val_loss} | TEACHER STUDENT VAL ACC = {val_accuracy*100}%')
        if best_acc < val_accuracy:
            best_acc = val_accuracy
            teacher_student_best_dict = teacher_student_model.student.state_dict()

    torch.save(teacher_student_best_dict, config.student_model_path)

if __name__ == "__main__":
    run()