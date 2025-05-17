import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
