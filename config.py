teacher_model_path = "model/teacher.bin"
student_model_path = "model/student.bin"

input_data = "input/"

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

cifar10_mean = (0.4913997551666284, 0.48215855929893703, 0.4465309133731618)
cifar10_std = (0.24703225141799082, 0.24348516474564, 0.26158783926049628)


Epochs = 50
Batch_Size = 128
lr = 5e-3
warmup_stes = 0.2

in_channels = 3
out_channels = 64
num_conv = 6
num_classes = 10
dropout = 0.2

temperature = 3
teachers_input_student_ratio = 10
soft_and_hard_ratio = 10
include_t_n_s_mid_loss = False