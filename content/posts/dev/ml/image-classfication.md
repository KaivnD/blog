---
title: "使用Pytorch训练一个图像分类器"
date: 2022-06-22T14:55:02+08:00
tags: ["pytorch", "ml", "image-classfication"]
summary: 训练一个图片分类器
draft: true
---

## 逻辑编程

当我们谈到编程的时候，往往都是指程序员根据需求和一些设定来设计组织业务**逻辑**、**流程**的这么一个过程。
侧重点是逻辑，流程，这么样的一个程序往往都是完成一些特定的任务，并且是在特定条件下才可以执行的程序，
从而得到一种输入数据到输出结果的固定映射。

## 数据编程

然而，机器学习正好相反，解题过程变成了：设定一个**模型**，这个模型里有很多**参数**，这些参数决定了这个模型 **_对于指定的输入，
将会做出怎样的输出。_**

有了这个模型，再加上一系列 _标注_ 了其对应的类别的 _数据_，称为**数据集**就可以建立训练过程了。

- 首先，我们依次把一系列数据中的的每一项给模型“看”，“告诉”模型这一项数据是什么类别，这个过程可以成为学习。
- 然后，经过这一次的学习，模型参数被**校准**一次，使用一些特定的办法，让我们可以知道调校后的模型和我们**设想**的模型**差别**有多大，来定义模型的**优劣程度**。
- 接着，我们需要一种**优化算法**，找到最佳参数，以最小化损失函数。
- 最后，再回到开头，循环这个过程。

{{< mermaid >}}
flowchart LR
Data(获取数据) --> Update(校准模型)
Update --> Check(验证模型)
Check --> Data
{{< /mermaid >}}

这个过程叫做训练，我们会写一个程序来训练我们的模型，这个程序叫做训练程序，这样的过程可以看作 _以数据来编程_。这种类型的学习也叫监督学习，本文讨论的范畴都是监督学习。

## 基本概念

上述过程中，有几个概念，在此说明一下:

- **模型**：抽象的来说，就是表达输入输出之间的关系映射，也是这整个事情的目标。
  具体点来说，可以说模型是一种特定的神经网络，比如说，常在计算机视觉方向应用的`ResNet`、`MobileNet`等。
  结合上下文来说，这里说的模型指，一个能够输入任意图，对于训练目标类别进行判别，并输出这张图可能的类别判别结果的神经网路。

- 参数：参数是模型内部的设定，参数的变动会影响输入数据在模型中的流动过程，从而影响最终输出结果。

- **数据集**：一般分为**训练集**（train）、**验证集**（validation）和**测试集**（test），都是已标记的数据。训练集用来训练模型，进行调模型参数，验证集用于评估调过参的模型，测试集是不参与训练过程，模型完全没有见过的数据，用于评估模型性能以及正确率。训练集相当于是平时考试，不断在让模型适应更多的情况，验证集相当于是期末考试，对于平时考试成绩很好，到了期末考试成绩却不理想的这种情况，称为**过拟合**，也即是，训练集的表现不能推广到验证集。

- 标注：在监督学习中，训练数据是需要被正确标记的，这些数据会影响训练结果。

- 校准：指训练过程中，模型“看了”，人为标记好类别的图像，相当于人在告诉模型，这个图像是什么图。
  这个图就是一个输入，图的类别是一个输出，然后根据这个输入和输出来调整模型的参数，使得模型可以适应这一个图像。

- **优劣程度**：每经过一次的学习，要对模型的学习成果打一个分儿，定义一个函数，来评价这一次学习之后，是进步了还是退步了，这个函数叫做**目标函数**，也叫做**损失函数**，函数值越低表示离模型设定的目标越接近。损失值用于指导后续的优化过程，目标是把损失降低。

- **优化算法**：用于在训练过程中，为模型找到最佳参数，以最小化损失函数。在深度学习中，大部分优化算法基于一个基本方法---**梯度下降**。梯度下降好比是在下山的过程中，寻找一条最快的下山路径。

- 学习率：学习训练这个过程是一步一步来的，每一步都根据优化函数调参，学习率会影响模型调参，学习率越高，调参幅度越大。

接下来看具体实现过程。

## 实现过程

### 迁移学习

根据上述内容，要训练一个图像分类器，我们需要以首先准备好数据集，要训练好一个模型，数据集采集非常关键，不光要保证质量，还要保证数量。数量，是训练过程的一个关键，比如，[Fashion-MNIST]()数据集有 6 万张衣服裤子鞋子等 10 个类别的图片，[ImageNet]()数据集有超过 1000 万的图片，包括 1000 类的物体。

显然，这种量级的数据集的收集和标记都是非常消耗时间与金钱的。还好，要达成我们设定的目标，还有另一个方法，这一类方法叫做**迁移学习**，这种方法可以以经过大量数据训练后的模型作为起点，使用相对来说数量较少的数据集来接着训练，以实现将源数据集学到的知识，迁移到目标数据集，即使源数据集与目标数据集的目标无关，源数据集训练过的模型也可能会在目标数据集提取更通用的图像特征，这有助于识别边缘、纹理、形状和对象组合等。

{{< mermaid >}}
flowchart  
 OriginModel(源模型) -.-> TargetModel(目标模型)
OriginData(源数据集) --> OriginModel
TargetData(目标数据集) --> TargetModel
{{< /mermaid >}}

接下来我们选用 `resnext101_32x8d` 作为预训练模型，使用 2200 张已标记的建筑图片作为数据集进行训练，此案例标注的内容是，建筑物图片主体的拍摄角度正不正，可以联系我获取已标注的数据集。

### 数据集

本案例数据集是文件夹形式，结构如下：

```bash
facade-reading-dataset/
└─ 角度正/
   ├─ 0/
   │  ├─ ... # n 张拍摄角度不正的图片
   │  └─ *.jpg
   └─ 1/
      ├─ ... # n 张拍摄角度正的图片
      └─ *.jpg
```

首先尝试使用 Pytorch 的方式加载这一套数据集

```python
# -*- coding:utf-8 -*-
from torchvision.datasets import ImageFolder

train_data_dir = './facade-reading-dataset/角度正'
# 通过ImageFolder这个class可以把整个文件夹按照文件夹分的分类读进来，得到一个数据集对象
dataset = ImageFolder(train_data_dir)
```

也如你所见，这套数据集里的图片大小尺寸参差不齐，我们需要对这套图片进行**预处理**，标准化训练数据，来满足训练的需要。
接下来我们来了解一下，训练需要什么样的数据。

#### 张量

我们的图片要被转换成一种叫**张量**(tensor)的数据结构，tensor 这个概念，听起来陌生，但用起来却很熟悉，本质上就是具有**多个维度的数组**。

- 有零个维度的在数学上叫**标量**，如

```py
3.2
```

- 有一个维度的在书叙述叫**向量**，如

```py
[4.1, 3.2, 6.8]
```

- 有两个维度的在数学上叫**矩阵**，如

```py
[[ 0,  1,  2,  3],
 [ 4,  5,  6,  7],
 [ 8,  9, 10, 11]]
```

- 有两个以上维度的在数学上没有特别的名称，如

```py
[[[0.0795, 0.6859, 0.7714],
  [0.4406, 0.0035, 0.8233],
  [0.8697, 0.7141, 0.1750],
  [0.3817, 0.7805, 0.6938]],

  [[0.8262, 0.5479, 0.9715],
  [0.6530, 0.8996, 0.6254],
  [0.4344, 0.1232, 0.7419],
  [0.3087, 0.7006, 0.9890]]]
```

tensor 这个概念，可以扩展到更高的维度，他们的统称叫做**张量**。

那么回到刚才的问题，这个张量和我们的数据集中的图片应该是什么关系呢？我们知道，图片是有一个个像素组成的，
而像素是具有一定的颜色值的，并且图片是一张二维的平面，每个像素点的位置是可以用坐标来表示的，那么这个图片，
就可以形成一个二维矩阵。如果用 RGB 三个通道来解析像素点的颜色值，那么，可以得到三个二维矩阵，
用这三个二维矩阵可以表示一张图，这一张图像就被数字化，比如说，下面这张图：

{{< columns >}}
<--->
![Bruder Klaus Chapel](/posts/dev/ml/imgs/building.jpg)
<--->
{{< /columns >}}

通过 RGB 三个通道取值之后，使用纯数值的形式看这张图就会变成这样：

{{< columns >}}
![Bruder Klaus Chapel Red Channel](/posts/dev/ml/imgs/building_r.jpg)
Red Channel
<--->
![Bruder Klaus Chapel Green Channel](/posts/dev/ml/imgs/building_g.jpg)
Green Channel
<--->
![Bruder Klaus Chapel Blue Channel](/posts/dev/ml/imgs/building_b.jpg)
Blue Channel
{{< /columns >}}

RGB 取值范围是`[0, 255]`，这个数值不便于计算，所以我们除了要统一图像尺寸外，还需要做标准化处理，如下我们使用 Pytorch 提供的 transforms 工具来对图像进行预处理。

<!-- TODO 解释这段代码 -->

```python
# -*- coding:utf-8 -*-
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

train_data_dir = './facade-reading-dataset/角度正'
image_size = 224

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
image_resize = int(image_size * (256 / 224)), int(image_size * (256 / 224))

image_transform = T.Compose([
    T.Resize(image_resize),
    T.CenterCrop(image_size),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
])

dataset = ImageFolder(root=train_data_dir, transform=image_transform)

classes_cnt = len(dataset.classes)
```

然后，根据一定的比例对数据集进行切分成三份，一个训练集，一个验证集，一个测试集。

<!-- TODO 解释这段代码 -->

```python
train_split, validation_split, test_split = 0.7, 0.2, 0.1

# 计算每个训练集的
train_size = int(len(dataset) * train_split)
validation_size = int(len(dataset) * validation_split)
test_size = len(dataset) - train_size - validation_size
```

使用 pytorch 提供的随机分割方法，将数据集随机分成三份。

```python
from torch.utils.data import random_split
random_seed = 330

train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size], generator=torch.Generator().manual_seed(random_seed))
```

最后，我们需要给训练程序提供一个数据集的批量加载器，稍后的训练程序就用这个来同时加载图像数据和标签。

<!-- TODO 解释这段代码 -->

```python
from torch.utils.data import DataLoader

batch_size = 16
data_loading_workers = 2

train_loader = DataLoader(
    train_dataset, batch_size=batch_size,
    num_workers=data_loading_workers, pin_memory=True,
)

validation_loader = DataLoader(
    validation_dataset, batch_size=batch_size,
    num_workers=data_loading_workers, pin_memory=True,
)
```

{{< details "合在一起" >}}

```python
# -*- coding:utf-8 -*-
import torch
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader

train_data_dir = './facade-reading-dataset/角度正'
image_size = 224
train_split, validation_split, test_split = 0.7, 0.2, 0.1
random_seed = 330
batch_size = 16
data_loading_workers = 2

image_resize = int(image_size * (256 / 224)), int(image_size * (256 / 224))

# 使用RGB通道的均值和标准差，以标准化每个通道
normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
image_transform = T.Compose([
    T.Resize(image_resize),
    T.CenterCrop(image_size),
    T.ToTensor(),
    normalize
])

dataset = ImageFolder(root=train_data_dir, transform=image_transform)
classes_cnt = len(dataset.classes)

train_size = int(len(dataset) * train_split)
validation_size = int(len(dataset) * validation_split)
test_size = len(dataset) - train_size - validation_size

train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size], generator=torch.Generator().manual_seed(random_seed))

train_loader = DataLoader(
    train_dataset, batch_size=batch_size,
    num_workers=data_loading_workers, pin_memory=True,
)

validation_loader = DataLoader(
    validation_dataset, batch_size=batch_size,
    num_workers=data_loading_workers, pin_memory=True,
)
```

{{< /details >}}

### 训练

训练过程一开始，先确定训练使用的设备是 GPU 还是 CPU

```python
import torch

# 让设备是cuda(NVIDIA GPU) 如果 cuda 可用的话，否则就用cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

从 torch hub (vision:v0.10.0) 加载预训练模型，模型架构是 `resnext101_32x8d`，指定`pretrained=True`以自动下载预训练的模型参数

```python
hub_repo = "pytorch/vision:v0.10.0"
model_arch = "resnext101_32x8d"

model = torch.hub.load(hub_repo, model_arch, pretrained=True)
```

我们采用**迁移学习**常用的一种手段，**微调** (Fine tune) 
修改源模型的fc层，并初始化随机参数

```python
import torch.nn as nn
from dataset import classes_cnt

model.fc = nn.Linear(model.fc.in_features, classes_cnt)
nn.init.xavier_uniform_(model.fc.weight)
model = model.to(device)
```

再模型参数中选择参与训练的参数

```py
params = [param for name, param in model.named_parameters() 
            if name not in ["fc.weight", "fc.bias"]]
```

定义优化函数为随机梯度下降
定义损失函数为交叉熵函数
定义学习率调整方法为按照间隔调整学习率

```py
import torch.optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

momentum = 0.9
initial_learning_rate = 5e-5
weight_decay = 1e-3

optimizer = torch.optim.SGD([{'params': params,}, {'params': model.fc.parameters(), 'lr': initial_learning_rate * 10}], 
                            lr=initial_learning_rate, momentum=momentum, weight_decay=weight_decay)

criterion = nn.CrossEntropyLoss().cuda()
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
```

定义迭代循环 (Epoch)，每迭代一次，代表训练集和验证集全部过一遍，模型进入下一代

```py
start_epoch = 0
epochs = 30

for epoch in range(start_epoch, epochs):
    print("--------------------------TRAINING--------------------------------")
```

接着上面的循环，进入训练模式，加载训练数据集

```py
from dataset import train_loader

# --------------------------接上面的Epoch循环--------------------------------")
    model.train()
    for i, (images, targets) in enumerate(train_loader):
        # 遍历每一组训练集的图片和标签

        # 将tensor转到当前设备
        images = images.to(device)
        targets = targets.to(device)

        # 传递图像输入给模型预测这个图像是属于哪个分类
        output = model(images)
        # 根据实际分类和预测分类使用损失函数计算损失
        loss = criterion(output, target)
```

计算准确率

```py
# --------------------------接上面的train_loader循环--------------------------------")
        # 
        prediction = torch.max(output, 1)[1]
        train_correct = (prediction == target).sum()
        train_acc = (train_correct.float()) / batch_size
```

打印训练日志

```py
print_freq = 2
# --------------------------接上面的train_loader循环--------------------------------")
        if i % print_freq == 0:
            print("[Epoch {:02d}] ({:03d}/{:03d}) | Loss: {:.18f} | ACC: {:.2f} %".format(epoch, i + 1, 
                                                            len(train_loader), loss.item(), train_acc * 100))
```

反向传播

```py
# --------------------------接上面的train_loader循环--------------------------------")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

进入评估模式

```py
# --------------------------接上面的Epoch循环--------------------------------")
    print("--------------------------VALIDATION-----------------------------")
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(validation_loader):
            # 遍历每一组验证集的图片和标签

            # 将tensor转到当前设备
            images = images.to(device)
            targets = targets.to(device)

            output = model(images)
            loss = criterion(output, targets)

            prediction = torch.max(output, 1)[1]
            val_correct = (prediction == targets).sum()
            val_acc = (val_correct.float()) / batch_size
                
            if i % print_freq == 0:
                print("[Epoch {:02d}] ({:03d}/{:03d}) | Loss: {:.18f} | ACC: {:.2f} %".format(epoch, i + 1, len(validation_loader), loss.item(), val_acc * 100))
```

这一代的训练结束，收尾工作，有两个：

1. 更新学习率
2. 保存这一代训练模型的参数

```py
import os

# --------------------------接上面的Epoch循环--------------------------------")
    scheduler.step()

    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, "checkpoint.pth")
    state = {
        'epoch': epoch + 1,
        'arch': model_arch,
        'hub_repo': hub_repo,
        'state_dict': model.state_dict(),
        'classes_cnt': classes_cnt
    }

    torch.save(state, filename)
```

