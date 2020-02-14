---


---

<h1 id="线性回归">线性回归</h1>
<h2 id="线性回归从零实现">线性回归从零实现</h2>
<h3 id="总体思路">总体思路</h3>
<p>实现的总体思路为：<br>
（1）生成（导入）数据<br>
（2）定义函数读取数据<br>
（3）初始化模型参数<br>
（4）定义模型（实现前向传播）<br>
（5）定义损失函数（计算损失）<br>
（6）模型训练</p>
<h3 id="生成数据">生成数据</h3>
<pre><code>num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs,
                    dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                    dtype=torch.float32)
</code></pre>
<p>代码解释：我们在这里生成一个拥有1000个样本的数据集，每个样本有两个属性，样本按照行排列的方式堆叠。使用权重与偏差<code>true_w</code>以及<code>true_b</code>生成样本的标注，并通过最后一行代码加上服从高斯分布的噪声。</p>
<h3 id="数据读取">数据读取</h3>
<p>由于我们在这里采用的是mini-batch的方式进行梯度下降，因此我们在此定义一个函数以便于我们在训练中每次能随机抽取mini-batch size大小的样本。代码如下：</p>
<pre><code>def data_iter(batch_size, features, labels):
    num_examples = len(features)  #获取样本总数量
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield  features.index_select(0, j), labels.index_select(0, j)
</code></pre>
<p>代码解释：通过<code>random.shuffle(indices)</code>将样本的索引随机打乱，这样在后面我们通过循环每次切取batch_size大小的索引时就相当于从数据集中随机挑选样本。程序中有个细节，就是最后一行使用了yield关键字，关于yield的使用我在网上找到了一篇感觉还不错的博客。<a href="https://liam.page/2017/06/30/understanding-yield-in-python/">Python 中的黑暗角落（一）：理解 yield 关键字</a>能看懂当然最好，看不懂也没关系（我没有看懂…），在博客里面给了一个简单粗暴的理解办法：</p>
<blockquote>
<ul>
<li>在函数开头加入result[]</li>
<li>将每个 <code>yield</code> 表达式 <code>yield expr</code> 替换为<code>result.append(expr)</code></li>
<li>末尾加上return result</li>
</ul>
</blockquote>
<p>在这里，相当于我们利用<code>yield</code>与循环生成了一个可以遍历的迭代器，在每次循环中，会从数据集里面选取相应的样本。由于数据是按行堆叠的，因此<code>features.index_select(0, j)</code>中参数表示按照行方向（dim=0）选取对应的样本，j中存储了要选取的样本对应的索引。</p>
<h3 id="参数初始化">参数初始化</h3>
<pre><code>w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
</code></pre>
<pre><code>w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True) 
</code></pre>
<h3 id="定义模型">定义模型</h3>
<pre><code>def linreg(X, w, b):  
    return torch.mm(X, w) + b
</code></pre>
<p>解释：torch.mm作矩阵乘法</p>
<h3 id="定义损失函数">定义损失函数</h3>
<pre><code>def squared_loss(y_hat, y):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    return (y_hat - y.view(y_hat.size())) ** 2 / 2
</code></pre>
<h3 id="定义优化算法">定义优化算法</h3>
<pre><code>def sgd(params, lr, batch_size):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data
</code></pre>
<p>注意这里的梯度要除以batch_size，因为我们最后以平均梯度对参数进行更新。</p>
<h3 id="模型训练">模型训练</h3>
<pre><code>lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    # 和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数

        # 不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
</code></pre>
<p><code>l.backward()</code>可以自动实现对参数求梯度，但不能对参数进行更新，因此我们调用<code>sgd([w, b], lr, batch_size)</code>更新参数，还要注意的是在更新之后要对参数清零，否则下一个批次计算的梯度会与之前的累加。</p>
<p><strong>参考资料</strong>：<br>
<a href="https://liam.page/2017/06/30/understanding-yield-in-python/">Python 中的黑暗角落（一）：理解 yield 关键字</a><br>
<a href="https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.3_linear-regression-pytorch">《动手学深度学习》pytorch实现</a></p>

