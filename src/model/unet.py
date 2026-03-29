import torch
import torch.nn as nn


class DoubleConv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		return self.net(x)


class Down(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

	def forward(self, x): 
		return self.pool_conv(x)


class Up(nn.Module):
	def __init__(self, in_ch, out_ch, bilinear=True):
		super().__init__()
		if bilinear:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
			self.conv = DoubleConv(in_ch, out_ch)
		else:
			self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
			self.conv = DoubleConv(in_ch, out_ch)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		# pad if needed
		diffY = x2.size(2) - x1.size(2)
		diffX = x2.size(3) - x1.size(3)
		if diffY or diffX:
			x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)


class OutConv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

	def forward(self, x):
		return self.conv(x)


class UNet(nn.Module):
	def __init__(self, in_channels=1, out_channels=1, features=(64, 128, 256, 512), bilinear=True):
		super().__init__()
		self.in_conv = DoubleConv(in_channels, features[0])
		self.downs = nn.ModuleList()
		for i in range(len(features) - 1):
			self.downs.append(Down(features[i], features[i + 1]))

		self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

		self.ups = nn.ModuleList()
		rev_feats = list(reversed(features))
		up_in_ch = features[-1] * 2
		for feat in rev_feats:
			self.ups.append(Up(up_in_ch, feat, bilinear=bilinear))
			up_in_ch = feat * 2

		self.out_conv = OutConv(features[0], out_channels)

	def forward(self, x):
		x1 = self.in_conv(x)
		skip_connections = [x1]
		x = x1
		for down in self.downs:
			x = down(x)
			skip_connections.append(x)

		x = self.bottleneck(x)

		# reverse skip connections for upsampling
		skip_connections = list(reversed(skip_connections))

		for idx, up in enumerate(self.ups):
			skip = skip_connections[idx]
			x = up(x, skip)

		logits = self.out_conv(x)
		return logits


if __name__ == '__main__':
	# quick smoke test
	model = UNet()
	x = torch.randn(2, 1, 64, 64)
	y = model(x)
	print('UNet output shape:', y.shape)