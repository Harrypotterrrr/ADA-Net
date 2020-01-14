import torch.nn as nn
import torch.nn.functional as F

class MetaLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(MetaLinear, self).__init__(*args, **kwargs)
        
    def forward(self, x, inner_lr=None):
        if inner_lr is None:
            return F.linear(x, self.weight, self.bias)
        else:
            return F.linear(x, self.weight.data-inner_lr*self.weight.grad,
                            bias=self.bias-inner_lr*self.bias.grad if self.bias is not None else None)
           
class MetaConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(MetaConv2d, self).__init__(*args, **kwargs)
        
    def forward(self, x, inner_lr=None):
        if inner_lr is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(x, self.weight.data-inner_lr*self.weight.grad,
                            bias=self.bias-inner_lr*self.bias.grad if self.bias is not None else None,
                            stride=self.stride, padding=self.padding, dilation=self.dilation,
                            groups=self.groups)

class MetaConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super(MetaConvTranspose2d, self).__init__(*args, **kwargs)
        
    def forward(self, x, inner_lr=None, output_size=None):
        # type: (Tensor, Optional[List[int]]) -> Tensor
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')
        output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size)
        if inner_lr is None:
            return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
                                      output_padding, self.groups, self.dilation)
        else:
            return F.conv_transpose2d(x, self.weight.data-inner_lr*self.weight.grad,
                                      bias=self.bias-inner_lr*self.bias.grad if self.bias is not None else None,
                                      stride=self.stride, padding=self.padding, output_padding=output_padding,
                                      groups=self.groups, dilation=self.dilation)        
       
class MetaBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super(MetaBatchNorm1d, self).__init__(*args, **kwargs)
            
    def forward(self, x, inner_lr=None):
        self._check_input_dim(x)
        
        if inner_lr is None:
            if self.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.momentum
    
            if self.training and self.track_running_stats:
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum
    
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                                self.training or not self.track_running_stats,
                                exponential_average_factor, self.eps)
        else:
            return F.batch_norm(x, self.running_mean, self.running_var,
                                self.weight.data-inner_lr*self.weight.grad if self.weight is not None else None,
                                self.bias.data-inner_lr*self.bias.grad if self.bias is not None else None,
                                training=True, momentum=0., eps=self.eps)

class MetaBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(MetaBatchNorm2d, self).__init__(*args, **kwargs)
            
    def forward(self, x, inner_lr=None):
        self._check_input_dim(x)
        
        if inner_lr is None:
            if self.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.momentum
    
            if self.training and self.track_running_stats:
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum
    
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                                self.training or not self.track_running_stats,
                                exponential_average_factor, self.eps)
        else:
            return F.batch_norm(x, self.running_mean, self.running_var,
                                self.weight.data-inner_lr*self.weight.grad if self.weight is not None else None,
                                self.bias.data-inner_lr*self.bias.grad if self.bias is not None else None,
                                training=True, momentum=0., eps=self.eps)


class MetaSequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super(MetaSequential, self).__init__(*args, **kwargs)
    
    def forward(self, x, inner_lr=None):
        for module in self._modules.values():
            if "Meta" in str(module):
                x = module(x, inner_lr=inner_lr)
            else:
                x = module(x)
        return x
