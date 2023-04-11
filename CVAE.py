import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):
    def __init__(self, feature_size, hidden_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size

        # encode
        self.fc1 = nn.Linear(feature_size + class_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + class_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c):  # Q(z|x, c)
        """
        x: (bs, feature_size)
        c: (bs, class_size)
        """
        inputs = torch.cat([x, c], 1)  # (bs, feature_size+class_size)
        h1 = self.elu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    @staticmethod
    def one_hot(labels, class_size, device):
        targets = torch.zeros(labels.shape[0], class_size)
        for i, label in enumerate(labels):
            targets[i, int(label.item())] = 1
        return targets.to(device)

    def decode(self, z, c):  # P(x|z, c)
        """
        z: (bs, latent_size)
        c: (bs, class_size)
        """
        inputs = torch.cat([z, c], 1)  # (bs, latent_size+class_size)
        h3 = self.elu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        mu, log_var = self.encode(x.view(-1, x.shape[1]), c)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, c), mu, log_var


def loss_function(x, x_recon, recon_embed, embed, mu, log_var, cof):
    RCE = torch.sqrt(F.mse_loss(x, x_recon))
    BCE = F.l1_loss(recon_embed, embed.view(-1, embed.shape[1]), reduction='mean')  # 改成了L1损失
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return RCE + cof * (BCE + KLD)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function_positive(x, x_recon, recon_embed, embed, mu, log_var, y, cof):
    square = torch.square(x - x_recon) * (1 - 2 * y).unsqueeze(2)
    RCE = torch.sqrt(torch.mean(square))

    absolute = torch.abs(recon_embed - embed.view(-1, embed.shape[1])) * (1 - 2 * y).unsqueeze(2)
    BCE = torch.mean(absolute)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return RCE + cof * (BCE + KLD)
