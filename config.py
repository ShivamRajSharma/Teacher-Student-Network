teacher_model_path = "model/teacher.bin"
student_model_path = "model/student.bin"

input_data = "input/"

# cifar100_mean = (0.5071, 0.4867, 0.4408)
# cifar100_std = (0.2675, 0.2565, 0.2761)

cifar100_mean = (0, 0, 0)
cifar100_std = (1, 1, 1)

Epochs = 50
Batch_Size = 128
lr = 1e-3
warmup_stes=0.2

in_channels=3
out_channels = 64
num_conv = 6
num_classes=100
dropout = 0.2


teachers_input_student = 0