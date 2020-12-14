import torch

def make_false_data(N, F_bin, T):
    mock = torch.rand([N, F_bin, T], dtype=torch.float32)
    mock2 = mock + torch.rand([N, F_bin, T], dtype=torch.float32) * 1 - 0.5
    mock = torch.stack([mock, mock2], dim=1)
    return mock
