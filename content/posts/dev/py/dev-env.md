---
title: "Python 开发环境安装"
date: 2022-06-16T00:58:24+08:00
keywords: ["python", "miniconda", "conda", "vscode"]
tags: ["python", "miniconda", "conda", "vscode"]
summary: Python 开发环境安装
---

{{< hint info >}}
**Tips**  
已安装开发环境，请前往[Python 起步](/posts/dev/py/starter)，开始学习。
{{< /hint >}}

## 本地

本地环境编程比较方便 debug，但 GPU 运算能力受本地机器限制。

### **VSCode**

VSCode 是微软开发的多语言多平台集成开发环境，编辑器选这一个就够用了，非必要不选别的编辑器。

{{< button href="https://code.visualstudio.com/Download" >}}官网下载{{< /button >}}

{{< hint info >}}
**Tips**  
安装过程附加任务勾选两个“通过 Code 打开”以及添加到 PATH
{{< /hint >}}

### **Miniconda**

是[Anaconda](https://www.anaconda.com)的 mini 版，仅保留核心功能，控制台作为交互界面。提供 Python 运行虚拟环境，这个虚拟环境是独立的运行环境，可以让你在不同工作项目的依赖独立，各管各的，互不影响。

#### 下载

点击下面按钮下载后双击安装，安装过程最好是啥都别改，一路到完成即可。

{{< button href="https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda" >}}清华镜像{{< /button >}}
{{< button href="https://docs.conda.io/en/latest/miniconda.html" >}}官网下载{{< /button >}}

#### 配置

- 设置环境变量

{{< tabs "uniqueid" >}}
{{< tab "Windows" >}}

## Windows

如果安装过程啥都没改的话，直接复制下面指令在 powershell 中执行，
如果修改了安装位置，修改变量`$loc`到安装目录（包含 miniconda3 的文件夹）。

```ps1
$loc = "%USERPROFILE%";
[Environment]::SetEnvironmentVariable("PATH", [Environment]::GetEnvironmentVariable("PATH", [EnvironmentVariableTarget]::User) + ";$loc\miniconda3;$loc\miniconda3\Scripts;$loc\Library\bin", [EnvironmentVariableTarget]::User);
```

{{< /tab >}}
{{< tab "MacOS" >}}

## MacOS

TODO
{{< /tab >}}

{{< /tabs >}}

- Shell 配置

{{< tabs "uniqueid" >}}
{{< tab "Windows" >}}

## Windows

1. 以管理员身份运行 powershell(Win+X A)

2. 设置执行策略，复制如下指令，执行后，输入 `Y` 确认。

```ps1
Set-ExecutionPolicy RemoteSigned
```

3. 执行 conda 配置 powershell 指令

```ps1
conda init powershell
```

4. 再次打开 powershell，`PS` 前面多了`(base)`，这就是进入了默认的`conda`环境，环境名叫`base`效果如下：

```bash
Windows PowerShell
版权所有 (C) Microsoft Corporation。保留所有权利。

尝试新的跨平台 PowerShell https://aka.ms/pscore6

加载个人及系统配置文件用了 1365 毫秒。
(base) PS C:\Users\KaivnD>
```

**注意** ：如果你的 powershell 没有变成这个样子，说明这一步失败了，会对后面的 vscode 调用 conda 有些影响

{{< /tab >}}
{{< tab "MacOS" >}}

## MacOS

TODO
{{< /tab >}}

{{< /tabs >}}

- 配置清华镜像（可选）

{{< hint info >}}
**Tips**  
由于某些原因，conda 下载包的时候网速会慢，所以，国内朋友最好还是设置一下。
{{< /hint >}}

1. 生成 conda 配置文件，执行如下指令，将会在用户目录下找到一个名为`.condarc`文件

```bash
conda config --set show_channel_urls yes
```

2. 在 vscode 中打开这个文件，复制如下内容到这个文件，然后保存这个文件。

```ini
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

3. 设置 pypi 镜像

在 powershell，执行如下指令，即可

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 虚拟环境

- **创建**

如下指令创建一个名为`env_name`的`python 3.9`的运行环境

```bash
conda create -n env_name python=3.9
```

- **进入**

如下指令进入名为`env_name`的虚拟环境，在进入这个环境后，
使用conda或者pip安装的包都只属于这个虚拟环境，进入之后，所执行python程序，
就会自动运行这个环境下的python版本以及调用这个环境下安装过的依赖。

```bash
conda activate env_name
```

- **退出**

如下指令可在`activate`执行之后，退出激活的虚拟环境。
如果在`base`环境下执行，会退出到没有虚拟环境的shell。

```bash
conda deactivate
```