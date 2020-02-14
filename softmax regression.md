---


---

<h1 id="softmax回归从零实现">softmax回归从零实现</h1>
<p>softmax回归同线性回归不同，它主要用于解决多分类问题，应用softmax回归，最终的输出相当于一个概率分布，即样本对应不同分类的概率。实现的总体思路与线性回归类似，不过其中有些细节不同，下面进行说明。</p>
<h3 id="softmax运算的实现">softmax运算的实现</h3>
<pre><code>def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制
</code></pre>
<p>X是我们模型的输出，每一行是一个样本对应分类的概率分布，因此我们在程序中指定dim=1，计算每一行指数求和，然后每一个元素除以该行的求和值得到结果。</p>
<h3 id="损失函数的定义">损失函数的定义</h3>
<pre><code>def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))
</code></pre>
<p>这里也有一个细节，即pytorch中gather的使用，也推荐一个博客：<a href="https://www.cnblogs.com/HongjianChen/p/9451526.html">Pytorch的gather用法理解</a><br>
在这里我们采用的也是交叉熵损失，不过此处的形式与二分类逻辑回归有所不同，是对各个样本属于标注分类的概率求对数后进行求和，<code>y.view(-1,1)</code>确保标注是一个列向量，<code>y_hat.gather(1, y.view(-1, 1)</code>语句得到模型计算出的每个样本对应于真实标注的概率。</p>
<h3 id="计算分类准确率">计算分类准确率</h3>
<pre><code>def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()
</code></pre>
<p><code>y_hat.argmax(dim=1)</code>得到每一行元素最大值对应的列索引，也就是该行对应的样本通过计算得到的类别，通过<code>y_hat.argmax(dim=1) == y</code>判断类别的输出是否准确产生一个布尔型的张量，<code>float()</code>将布尔值转为数值，通过<code>mean</code>得到准确率(<code>mean</code>在pytorch用于求平均值，这里计算出来的结果相当于1在整个张量中所占的比例，因此也就作为准确率)，<code>item</code>将类型转变为一个标量。</p>
<h3 id="模型训练">模型训练</h3>
<pre><code>num_epochs, lr = 5, 0.1

# 本函数已保存在d2lzh包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到


            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)
</code></pre>
<p><strong>参考资料</strong><br>
<a href="https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.6_softmax-regression-scratch">softmax回归的从零开始实现</a></p>

