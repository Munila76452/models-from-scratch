import torch
from torch import nn

class variationalAutoencoder(nn.Module):
  def __init__(self,int_dim,hidd_dim=20,z_dim=20):
    super().__init__()
    # encoder
    self.img_2_hidd = nn.Linear(int_dim,hidd_dim)
    self.hidd_2_mu = nn.Linear(hidd_dim,z_dim)
    self.hidd_2_sigma = nn.Linear(hidd_dim,z_dim)

    # decoder
    self.z_2_hidd = nn.Linear(z_dim,hidd_dim)
    self.hidd_2_img = nn.Linear(hidd_dim,int_dim)

    self.relu = nn.ReLU()
  def encoder(self,x):
    h = self.relu(self.img_2_hidd(x))
    mu = self.hidd_2_mu(h)
    sigma = self.hidd_2_sigma(h)
    return mu,sigma
  def decoder(self, z):
    h = self.relu(self.z_2_hidd(z))
    return torch.sigmoid(self.hidd_2_img(h))

  def forward(self,x):
    mu,sigma = self.encoder(x)
    epsilon = torch.rand_like(sigma)
    z_reparametrazed = mu + sigma*epsilon
    x_reconstructed = self.decoder(z_reparametrazed)
    return x_reconstructed,mu,sigma
if __name__ == "__main__":
  x = torch.randn(4,28*28)
  vae = variationalAutoencoder(int_dim=784)
  x_reconstructed,mu,sigma = vae(x)
  print(x_reconstructed.shape)
  print(mu.shape)
  print(sigma.shape)