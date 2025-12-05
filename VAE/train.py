import torch
from torch import nn
import torchvision.datasets as datasets
from tqdm import tqdm
from models import variationalAutoencoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

input_dim = 784
h_dim = 200
z_dim = 20
num_epoch = 10 
batch_size = 32
lr_rate = 3e-4 # Karpathy's constant

dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = variationalAutoencoder(input_dim, h_dim, z_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

loss_fn = nn.BCELoss(reduction="sum")

# Training
for epoch in range(num_epoch):
    loop = tqdm(dataloader) 
    for i, (x, _) in enumerate(loop):
        x = x.to(device).view(x.shape[0], input_dim)
        x_reconstructed, mu, logvar = model(x)
        reconstruction_loss = loss_fn(x_reconstructed, x)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = reconstruction_loss + kl_divergence

        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        loop.set_postfix(loss=loss.item())
save_image(x_reconstructed.view(batch_size, 1, 28, 28), "reconstructed.png")
print("all done")