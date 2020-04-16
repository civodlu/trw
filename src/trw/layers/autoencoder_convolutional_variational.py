import torch.nn as nn
import torch.nn.functional as F
import torch
from trw.layers import flatten, OpsConversion, crop_or_pad_fun


class AutoencoderConvolutionalVariational(nn.Module):
    """
    Variational convolutional autoencoder implementation

    See good reference:
        https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/

    """
    def __init__(self, cnn_dim, encoder, decoder, z_input_filters, z_output_filters):
        """

        Args:
            cnn_dim: the dimension of the input space. For example, 2 for 2D images or 3 for 3D image.
            encoder: the encoder, taking input X and mapping to intermediate feature space Y
            decoder: the decoder, taking input Z and mapping back to input space X. If the decoder output
                is not X shaped, it will be padded or cropped to the right shape
            z_input_filters: the number of filters of the last layer of the encoder. The bottleneck
                is implemented as a N-d convolution
            z_output_filters: the number of filters to be used for the bottleneck. The bottleneck
                is implemented as a N-d convolution
        """
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder
        self.z_input_filters = z_input_filters
        self.z_output_filters = z_output_filters

        ops_conv = OpsConversion(cnn_dim)

        # keep the mu & logvar spatial dimensions
        self.z_mu = ops_conv.conv_fn(z_input_filters, z_output_filters, kernel_size=1)

        # in the original paper (Kingma & Welling 2015, we
        # have a z_mean and z_var, but the problem is that
        # the z_var can be negative, which would cause issues
        # in the log later. Hence we assume that latent vector
        # has a z_mean and z_log_var component, and when we need
        # the regular variance or std_dev, we simply use
        # an exponential function
        self.z_logvar = ops_conv.conv_fn(z_input_filters, z_output_filters, kernel_size=1)

    def encode(self, x):
        n = self.encoder(x)
        assert n.shape[1] == self.z_input_filters, f'expecting {self.z_input_filters} filters ' \
                                                   f'for the encoder, got={n.shape[1]}'
        mu = self.z_mu(n)
        logvar = self.z_logvar(n)
        return mu, logvar

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(self.training, mu, logvar)
        recon = self.decoder(z)

        # make the recon exactly the same size!
        recon = crop_or_pad_fun(recon, x.shape[2:])
        assert recon.shape == x.shape, f'recon ({recon.shape}) and x ({x.shape}) must have the same shape.' \
                                       f'problem with the decoded!'
        return recon, mu, logvar

    @staticmethod
    def reparameterize(training, z_mu, z_logvar):
        """
        Use the reparameterization ``trick``: we need to generate a
        random normal *without* interrupting the gradient propagation.

        We only sample during training.
        """
        if training:
            # note that log(x^2) = 2*log(x); hence divide by 2 to get std_dev
            # i.e., std_dev = exp(log(std_dev^2)/2) = exp(log(var)/2)
            std = torch.exp(0.5 * z_logvar)
            eps = torch.randn_like(std)
            z = z_mu + eps * std
        else:
            z = z_mu

        return z

    @staticmethod
    def loss_function(recon_x, x, mu, logvar, recon_loss_name='BCE', kullback_leibler_weight=0.2):
        """
        Loss function generally used for a variational auto-encoder

        compute:
            reconstruction_loss + Kullback_Leibler_weight * Kullback–Leibler divergence((mu, logvar), gaussian(0, 1))

        Args:
            recon_x: the reconstructed x
            x: the input value
            mu: the mu encoding of x
            logvar: the logvar encoding of x
            recon_loss_name: the name of the reconstruction loss. Must be one of ``BCE`` (binary cross-entropy) or
                ``MSE`` (mean squared error)
            kullback_leibler_weight: the weight factor applied on the Kullback–Leibler divergence. This is to
                balance the importance of the reconstruction loss and the Kullback–Leibler divergence

        Returns:
            a 1D tensor, representing a loss value for each ``x``
        """
        if recon_loss_name == 'BCE':
            recon_loss = F.binary_cross_entropy(recon_x, x, reduction='none')
        elif recon_loss_name == 'MSE':
            recon_loss = F.mse_loss(recon_x,  x)
        else:
            raise NotImplementedError(f'loss not implemented={recon_loss_name}')
        recon_loss = flatten(recon_loss).mean(dim=1)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kullback_leibler = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kullback_leibler = flatten(kullback_leibler).mean(dim=1)

        return recon_loss + kullback_leibler * kullback_leibler_weight
