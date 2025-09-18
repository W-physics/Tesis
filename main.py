import torch

torch.cuda.is_available()

device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'

torch.set_default_device(device)

from backward_process.train_generate import TrainGenerate

def main():
    TrainGenerate()

main()