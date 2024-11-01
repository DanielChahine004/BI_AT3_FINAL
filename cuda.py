import torch

print("Available CUDA devices:")
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")




for i in range(torch.cuda.device_count()):
    torch.cuda.set_device(i)
    tensor = torch.randn(2, 2).cuda()
    print(f"Tensor on Device {i}: {tensor}")

    
