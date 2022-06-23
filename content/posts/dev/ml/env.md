---
title: "机器学习开发环境"
date: 2022-06-17T13:47:56+08:00
tags: ["python", "pytorch", "ml", "ai"]
summary: 机器学习开发环境
draft: true
---

{{< hint info >}}
**先决条件**

请先阅读如下两篇文章，再来阅读这篇文章。

[安装开发环境](/posts/dev/py/env)

[Python 起步](/posts/dev/py/starter)

{{< /hint >}}

## 环境配置

- 依次执行下面两条指令，创建并激活名为`torchenv`，python版本为3.9的虚拟环境。

```bash
conda create -n torchenv -y python=3.9
conda activate torch env
```

- 安装Pytorch

{{< tabs "uniqueid" >}}
{{< tab "Windows" >}}

## Windows

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

{{< /tab >}}
{{< tab "MacOS" >}}

## MacOS

```bash
conda install pytorch torchvision torchaudio -c pytorch
```
{{< /tab >}}
{{< /tabs >}}
