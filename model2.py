import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class MultiVAE(nn.Module):
    """
    Container module for Multi-VAE v2.
    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims
        if q_dims is not None:
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]
        self.p_dims[0] = self.q_dims[-1]//2
        self.encoder = self.mlp_layers(self.q_dims)
        self.decoder = self.mlp_layers(self.p_dims)
        # Last dimension of q- network is for mean and variance
        self.drop = nn.Dropout(dropout)
        self.apply(MultiVAE.init_weights)
        
    def mlp_layers(self, layer_dims):
        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.Tanh())
        return nn.Sequential(*mlp_modules)
    
    def forward(self, input):
        h = F.normalize(input)
        h = self.drop(h)
        h = self.encoder(input)
        
        mu = h[:, : self.p_dims[0]]
        logvar = h[:, self.p_dims[0] :]
        
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
    


    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.zeros_like(std).normal_(mean=0, std=0.01)
            return mu + epsilon * std
        else:
            return mu
        
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

def loss_function(recon_x, x, mu, logvar, anneal=1.0):
    # BCE = F.binary_cross_entropy(recon_x, x)
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD