import random
import time
import torch 
import torch.nn as nn


class Conv_Forward(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(Conv_Forward, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.fc(x)

class Teacher(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv, num_classes, dropout):
        super(Teacher, self).__init__()
        self.num_conv = num_conv
        self.conv_1 = nn.Conv2d(in_channels, out_channels, padding=1, kernel_size=3)
        self.activation = nn.LeakyReLU()
        self.drop_1 = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            *[
                Conv_Forward(out_channels, out_channels, dropout=dropout)
                for i in range(num_conv)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, 30),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(30, num_classes)
        )

    def forward(self, x):
        out_1 = self.drop_1(self.activation(self.conv_1(x)))
        out_2 = self.fc(out_1)
        out_3 = torch.mean(out_2, dim=(2, 3))
        out_4 = self.classifier(out_3)
        return out_1, out_2, out_3, out_4


class Student(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, dropout):
        super(Student, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout()
        )

        self.classifier = nn.Sequential(
            nn.Linear(out_channels, 30),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(30, num_classes)
        )

    
    def forward(self, x, teacher_out_1=None, teacher_out_2=None, teachers_input_student_ratio=10):
        out_1 = self.conv_1(x)

        if random.random() > 0.5:
            teacher_out_2 = None
        else:
            teacher_out_1 = None

        if (teacher_out_1 is not None) and (random.random() > teachers_input_student_ratio):
            out_2 = self.conv_2(teacher_out_1.detach())
        else:
            out_2 = self.conv_2(out_1)
        
        if (teacher_out_2 is not None) and (random.random() > teachers_input_student_ratio):
            out_3 = torch.mean(teacher_out_2.detach(), dim=(2, 3))
        else:
            out_3 = torch.mean(out_2, dim=(2, 3))

            
        out_4 = self.classifier(out_3)
        
        return out_1, out_2, out_3, out_4


class TeacherStudentNetwork(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        teacher_num_conv,
        teacher_weights_path,
        num_classes,
        dropout,
    ):
        super(TeacherStudentNetwork, self).__init__()
        self.teacher =  Teacher(
            in_channels=in_channels,
            out_channels=out_channels,
            num_conv=teacher_num_conv,
            num_classes=num_classes,
            dropout=dropout
        )

        self.student = Student(
            in_channels=in_channels,
            out_channels=out_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
        
        self.teacher.load_state_dict(torch.load(teacher_weights_path))

    def forward(self, x, teachers_input_student_ratio=10):
        with torch.no_grad():
            t_out_1, t_out_2, t_out_3, t_out_4 = self.teacher(x)
        s_out_1, s_out_2, s_out_3, s_out_4 = self.student(x, t_out_1, t_out_2, teachers_input_student_ratio)
        return (
            [t_out_1.detach(), t_out_2.detach(), t_out_3.detach(), t_out_4.detach()],
            [s_out_1, s_out_2, s_out_3, s_out_4]
        )


if __name__ == "__main__":
    img = torch.randn(2, 3, 32, 32)
    t = Teacher(3, 32, 4, 10, 0.2)
    s = Student(3, 32, 10, 0.2)
    start = time.time()
    out_1, out_2, out_3, out_4 = t(img)
    print(f"time taken for teacher = {time.time() - start}")
    start = time.time()
    out = s(img, out_1, out_2)
    print(f"time taken for student = {time.time() - start}")
    path = "model/teacher_model.bin"
    torch.save(t.state_dict(), path)
    t_s = TeacherStudentNetwork(3, 32, 4, path, 10, 0.2)
    start = time.time()
    x = t_s(img)
    print(f"time taken for teacher-student = {time.time() - start}")