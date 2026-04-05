import warnings
import torch

def my_cuda_init(verbose=False):

    if verbose:
        print("Specifying Device...")

    if torch.cuda.is_available():

        device = 'cuda'
        if verbose:
            print("Built with CUDA:", torch.version.cuda)
            print("Device count:", torch.cuda.device_count())
            print("GPU 0:", torch.cuda.get_device_name(0))

    else:

        device = 'cpu'
        warnings.warn("CUDA not available, using CPU!")

    return device

def printstats(device, message):

    if device == 'cuda':

        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        print(f"Stats at stage: {message}")

        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        print(f"GPU utilization: {util.gpu}%")
        #print(f"Memory utilization: {util.memory}%")

        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"Memory used: {mem_info.used / 1e9:.2f} GB")
        print(f"Memory total: {mem_info.total / 1e9:.2f} GB")
        print(f"Memory free: {mem_info.free / 1e9:.2f} GB")

        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        print(f"Temperature: {temperature}°C")

        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
        print(f"Power: {power:.2f} W")

        pynvml.nvmlShutdown()

    elif device == 'cpu':

        print("On CPU, so no GPU stats available")



