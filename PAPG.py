import torch
import torch.nn as nn
from torch.nn import functional as F

#####
def centre_crop(x, target):
    #cut for same length, return x_cut
    #x是下采样的中间层, target是上采样的中间层
    diff = x.shape[-1] - target.shape[-1]
    crop = diff // 2
    if diff == 0:
        return x
    if diff % 2 == 0:
        out = x[:, :, crop:-crop].contiguous()
    else:
        out = x[:, :, crop+1:-crop].contiguous()
    assert (out.shape[-1] == target.shape[-1])
    return out

class ConvLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, transpose=False):
        super(ConvLayer, self).__init__()
        self.transpose = transpose
        self.stride = stride
        self.kernel_size = kernel_size

        if self.transpose:
            self.filter = nn.ConvTranspose1d(n_inputs, n_outputs, self.kernel_size, stride, padding=kernel_size-1)
        else:
            self.filter = nn.Conv1d(n_inputs, n_outputs, self.kernel_size, stride)

        self.norm = nn.BatchNorm1d(n_outputs, momentum=0.01)
    
    def forward(self, x):
        m = self.filter(x)
        out = F.relu(self.norm(m))
        return out

    def get_input_size(self, output_size):
        if self.transpose:
            curr_size = output_size
        else:
            curr_size = (output_size - 1) * self.stride + 1
        
        curr_size = curr_size + self.kernel_size - 1

        if self.transpose:
            curr_size = ((curr_size - 1) // self.stride) + 1

        return curr_size
    
    def get_output_size(self, input_size):
        if self.transpose:
            curr_size = (input_size - 1) * self.stride + 1
        else:
            curr_size = input_size
        
        curr_size = curr_size - self.kernel_size + 1

        if not self.transpose:
            curr_size = ((curr_size - 1) // self.stride) + 1
        
        return curr_size

class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride):
        super(UpsamplingBlock, self).__init__()

        self.upconv = ConvLayer(n_inputs, n_inputs, kernel_size, stride, transpose=True)
        self.pre_shortcut_convs = ConvLayer(n_inputs, n_outputs, kernel_size, 1)
        self.post_shortcut_convs = ConvLayer(n_outputs + n_shortcut, n_outputs, kernel_size, 1)

    def forward(self, x, shortcut):
        #shortcut来自于对应级的下采样的中间层
        upsampled = self.upconv(x)
        upsampled = self.pre_shortcut_convs(upsampled)
        combined_down = centre_crop(shortcut, upsampled)
        combined_up = centre_crop(upsampled, combined_down)
        combined = self.post_shortcut_convs(torch.cat([combined_down, combined_up], dim=1))
        #截取等长上下采样同级中间层并拼接
        return combined
    
    def get_output_size(self, input_size):
        curr_size = self.upconv.get_output_size(input_size)
        curr_size = self.pre_shortcut_convs.get_output_size(curr_size)
        curr_size = self.post_shortcut_convs.get_output_size(curr_size)
        return curr_size

class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride):
        # n_inputs是输入的通道数,n_shortcut是中间层通道数,n_outputs是输出层通道数
        super(DownsamplingBlock, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.pre_shortcut_convs = ConvLayer(n_inputs, n_shortcut, kernel_size, 1)
        self.post_shortcut_convs = ConvLayer(n_shortcut, n_outputs, kernel_size, 1)
        self.downconv = ConvLayer(n_outputs, n_outputs, kernel_size, stride)

    def forward(self, x):
        shortcut = self.pre_shortcut_convs(x)
        out = self.post_shortcut_convs(shortcut)
        out = self.downconv(out)
        return out, shortcut
    
    def get_input_size(self, output_size):
        curr_size = self.downconv.get_input_size(output_size)
        curr_size = self.post_shortcut_convs.get_input_size(curr_size)
        curr_size = self.pre_shortcut_convs.get_input_size(curr_size)
        return curr_size

class AttentionBlock(nn.Module):
    def __init__(self, n_channels, kernel_size):
        super(AttentionBlock, self).__init__()
        self.conv = ConvLayer(n_channels, n_channels, kernel_size, 1)

    def forward(self, x_t, x_s):
        # x_s是uni, x_t是lookback声源音频
        out_t = self.conv(x_t)
        out_s = self.conv(x_s)
        out = out_t + out_s
        out = F.relu(out)
        out = self.conv(out)
        out = F.sigmoid(out)
        p1d = (0, x_t.size()[-1]-out.size()[-1])
        out = F.pad(out, p1d, 'constant', 0)
        out = x_s * out
        # out = x_t + out # 注意力是在这里用哈达玛积，我们改成了相加
        return out


class Attention_Waveunet(nn.Module):
    def __init__(self, snr, enroll, num_inputs, num_channels, num_outputs, kernel_size=5, strides=2):
        #num_inputs是输入级通道数,num_channels是每级的输入通道数,num_outputs是输出通道数等于输入通道数
        super(Attention_Waveunet, self).__init__()

        self.num_levels = len(num_channels)
        self.strides = strides
        self.kernel_size = kernel_size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.waveunets = nn.Module()
        self.waveunets.downsampling_blocks1 = nn.ModuleList()
        self.waveunets.downsampling_blocks2 = nn.ModuleList()
        self.waveunets.upsampling_blocks = nn.ModuleList()
        self.waveunets.attention_blocks = nn.ModuleList()

        for i in range(self.num_levels - 1):
            self.waveunets.attention_blocks.append(AttentionBlock(num_channels[i+1], kernel_size))

        for i in range(self.num_levels - 1):
            in_ch = num_inputs if i == 0 else num_channels[i]
            self.waveunets.downsampling_blocks1.append(DownsamplingBlock(in_ch, num_channels[i], num_channels[1+i], kernel_size, strides))

        for i in range(self.num_levels - 1):
            in_ch = num_inputs if i == 0 else num_channels[i]
            self.waveunets.downsampling_blocks2.append(DownsamplingBlock(in_ch, num_channels[i], num_channels[1+i], kernel_size, strides))

        for i in range(0, self.num_levels - 1):
            self.waveunets.upsampling_blocks.append(UpsamplingBlock(num_channels[-1-i], num_channels[-2-i], num_channels[-2-i], kernel_size, strides))
        
        self.waveunets.bottlenecks = ConvLayer(num_channels[-1], num_channels[-1], kernel_size, 1)
        self.waveunets.output_conv = nn.Conv1d(num_channels[0], num_outputs, 1)

        self.last_norm = nn.LayerNorm(enroll)
        self.last_linear = nn.Linear(enroll, enroll)
        self.snr = snr

    
    def forward(self, x_s, x_t):
        module=self.waveunets
        shortcuts = []
        out_s = x_s
        out_t = x_t

        for i in range(self.num_levels - 1):
            block1 = module.downsampling_blocks1[i]
            block2 = module.downsampling_blocks2[i]
            out_s, short = block1(out_s) # 对源音频进行编码
            out_t, _  = block2(out_t) # 对目标说话人音频进行编码
            attention_block = module.attention_blocks[i]
            out = attention_block(out_s, out_t) # 注意力加权
            shortcuts.append(short)

        out = module.bottlenecks(out) #下采样和上采样之间有一个瓶颈层

        for idx, block in enumerate(module.upsampling_blocks):
            out = block(out, shortcuts[-1 - idx]) #每一级上采样的输入层通道数是上一级输出层通道数

        out = module.output_conv(out)
        # if not self.training:  # At test time clip predictions to valid amplitude range
        #     out = out.clamp(min=-1.0, max=1.0)
        
        diff = x_s.shape[-1] - out.shape[-1]
        out = torch.cat([out, x_s[:, :, -diff:]], dim=2)
        out = self.last_linear(out)
        out = self.last_norm(out)
        out_fortest = out
        origin_l2 = torch.sum(x_s**2)
        power_perb = origin_l2 / (10.0**(30/10))
        perb_l2 = torch.sum(out**2)
        out = out * torch.sqrt((power_perb / perb_l2))
        out = out + x_s


        return out_fortest, out


        