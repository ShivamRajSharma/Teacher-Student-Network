teacher_model_path = "model/teacher.bin"
student_model_path = "model/student.bin"

input_data = "input/"

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

Epochs = 30
Batch_Size = 32
lr = 1e-3
warmup_stes=0.2

in_channels=3
out_channels = 32
num_conv = 4
num_classes=100
dropout = 0.2