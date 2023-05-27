import torch
import torch.nn as nn

class WinogradConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(WinogradConv2d, self).__init__()
        if kernel_size != 3:
            raise ValueError("Kernel size must be 3 for Winograd Convolution.")
        if stride != 1 or padding != 0 or dilation != 1 or groups != 1:
            raise ValueError("Only kernel_size=3, stride=1, padding=0, dilation=1, groups=1 is currently supported.")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if x.dim() != 4:
            raise ValueError("Input dimension must be 4.")
        if x.size(2) % 2 != 0 or x.size(3) % 2 != 0:
            raise ValueError("Input height and width must be even.")
        B, C, H, W = x.size()
        if H < 4 or W < 4:
            raise ValueError("Input height and width must be greater than or equal to 4 for Winograd Conv.")
        
        G = torch.Tensor([1, 0, 0,
                          0.5, 0.5, 0.5,
                          0.5, -0.5, 0.5,
                          0, 0, 1]).view(4, 3).to(x.device)
        if x.is_cuda:
            G = G.cuda()
        tiles = [x[:, :, i:i+4, j:j+4] for i in range(0, H-2, 2) for j in range(0, W-2, 2)]
        tiles = torch.stack(tiles, dim=0)
        tiles_transformed = torch.einsum('nbcwh,Gcw->nbGwh', (tiles, G))
        tiles_transformed = torch.einsum('nbchw,Gch->nbGw', (tiles_transformed, G))

        tiles_transformed = tiles_transformed.view(-1, C, 4, 4)
        y = self.conv(tiles_transformed)

        G_t = G.t()
        y = torch.einsum('nbcwh,Gcw->nbGwh', (y, G_t))
        y = torch.einsum('nbchw,Gch->nbGw', (y, G_t))

        y = y.view(-1, B, H//2, W//2)
        y = y.sum(dim=0)

        return y


if __name__ == "__main__":
    # Instantiate the module
    winograd_conv = WinogradConv2d(in_channels=1, out_channels=1, kernel_size=3)
    winograd_conv = winograd_conv.cuda()  # Comment this line out if you don't have a GPU
    
    # Create a mini-batch of images
    x = torch.randn(10, 1, 32, 32)
    x = x.cuda()  # Comment this line out if you don't have a GPU

    # Run the convolution
    y = winograd_conv(x)
    
    print(y.shape)  # Should print torch.Size([10, 1, 16, 16])
