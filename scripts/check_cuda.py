import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # Test a simple CUDA operation
    print("\nTesting CUDA tensor operation:")
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    z = x @ y  # Matrix multiplication
    end.record()
    
    # Waits for everything to finish running
    torch.cuda.synchronize()
    
    print(f"Operation time: {start.elapsed_time(end)} ms")
    print(f"Result tensor device: {z.device}")
else:
    print("CUDA is not available. Check your PyTorch installation.") 