#  pytorch实现卷积神经网络
##  相关公式
###  卷积层
- 输入图片大小$n\times n$
- 滤波器尺寸$f\times f$
- padding p
- stride s
输出大小为
$$\lfloor \frac{n+2p-f}{s}+1 \rfloor \times \lfloor \frac{n+2p-f}{s}+1 \rfloor $$
###  卷积层pytorch实现
使用nn.conv2d实现：
nn.conv2d(in_channels,out_channels,kernel_size,padding)
这里需要注意padding的使用，若padding=p，则表示在输入的高和宽两个方向**分别**填充p行（相当于一共填充了2p行），padding=(h,w)表示在两个方向分别填充h、w行。上面式子中的p指的不是在某个方向上一共填充的行数而是一侧填充的数目。
##  池化层
池化层的提出是为了**为了缓解卷积层对位置的过度敏感性**。
###  最大池化层的pytorch实现
nn.MaxPool2d(pooling_size,padding,stride)
pooling_size为池化窗口的尺寸，padding为填充的数目，stride为池化操作的步长。

