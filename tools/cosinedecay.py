# draw cosine decay learning rate curve
import torch
import matplotlib.pyplot as plt

A = torch.Tensor([1])
A.requires_grad=True

optimizer = torch.optim.Adam([A], lr=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
            T_0=100, eta_min=1e-7)
lrs = []
for i in range(100):
	lrs.append(optimizer.param_groups[0]['lr'])
	sched.step()
plt.xlabel('epoch')
plt.ylabel('learning rate')
plt.plot(lrs)
plt.show()
