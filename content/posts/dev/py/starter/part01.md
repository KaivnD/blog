---
title: "Python起步 01"
date: 2022-06-15T00:58:24+08:00
tags: ["python"]
summary: python快速起步
---

## Part 1 变量与序列

{{< hint info >}}
**注意**  
如果没有安装开发环境，请参照[安装开发环境](/posts/dev/py/env)
{{< /hint >}}

在一个程序里，变量是最核心的部分，它表示了数据在程序中的流动和变化。

### 1.1 变量声明与赋值

如下表达式，"=" 为赋值符号，等式左边为变量名称，右边为赋予变量的值。

{{< hint warning >}}
**注意**  
变量名称不能以数字开头，不能包含空格，不能是诸如 `if else and not for` 等关键词
{{< /hint >}}

常规赋值

```py
a = 1
b = 2
c = True
d = "Hello"
```

连续赋值

```py
e = f = 12
```

[元组](#123-元组)赋值

```py
g, h = 3, 6
```

### 1.2 序列

在 Python 中一共有四种序列，可以用来表示一系列的值

{{< details "扩展内容" >}}
1. 列表（List）是一种有序且可更改的集合，允许重复的元素。
2. 字典（Dictionary）是一个无序，可变和有索引的集合，没有重复的元素。
3. 元组（Tuple）是一种有序且不可更改的集合，允许重复的元素。
4. 集合（Set）是一个无序且无索引的集合，没有重复的元素。
{{< /details >}}

#### 1.2.1 列表

列表是 Python 中非常常用的序列表示，可以通过如下方法创建列表

```py
alist = [1, 2, 3, 4, 5, 6]
blist = ["a", "b", "c", "d"]
```

**新增**

```py
# 在列表的末尾新增一个数据
alist.append(7)
# 输出： [1, 2, 3, 4, 5, 6, 7]

# 在列表某个位置插入数据，第一个参数是位置，第二个参数是要插入的值
alist.insert(0, 0)
# 输出： [0, 1, 2, 3, 4, 5, 6, 7]
```

**删除**

```py
# 删除位置3的元素，del是python的关键字，作用是删除某个变量
del alist[3]
# 输出： [0, 1, 2, 4, 5, 6, 7]

# 删除列表中的值是2的元素，如果有多个值为2，那么删掉靠前的一个
alist.remove(2)
# 输出： [0, 1, 4, 5, 6, 7]
```

**取值**

因为列表是有序的，所以可以按照顺序对列表中的元素取值

```py
# 使用方括号操作符对列表变量进行索引，获取此位置的值
alist[0]
# 输出： 1

# 使用冒号可以提取某个范围的值得到另一个列表
alist[1:3]
# 输出： [2, 3, 4]

# 冒号前后都可以省略比如
alist[1:] # 表示提取从第一个开始一直到最后一个
# 输出： [2, 3, 4, 5, 6]
```

**修改**
赋值

```py
# 2号位置赋值为9
alist[2] = 9
# 输出： [0, 1, 9, 5, 6, 7]

```

#### 1.2.2 字典

字典是一种映射关系，由多个键值对（一对数据，反应名称到值的映射关系，就像是查字典一样，可以根据键名，查到相应的键值）组成，
通常用来创建一些特定结构的数据

```py
# 字典可定义一组有映射关系的数据
building = {
    'name': 'xx大楼',
    'type': '公建',
    'height': 99
}
```

字典虽然是无序的，但是列表是键值对的映射，所以可以通过索引键名取值，也可以赋值

**新增**

```py
# 也可以直接设置一个新的键值对，值可以是任意类型，
# 这里将'floors'设置为一个列表，列表里有两个floor
# 这里的floor又是一个新的字典
building['floors'] = [
    {
        'name': 'L0',
        'height': 3
    },
    {
        'name': 'L1',
        'height': 4.5
    }
]
# 输出： {'name': 'xx大楼', 'type': '公建', 'height': 72, 'floors': [{'name': 'L0', 'height': 3}, {'name': 'L1', 'height': 4.5}]}
```

**删除**
```py
del building['name']
del building['floors']

{'type': '公建', 'height': 99}
```

**取值**
```py
building['name']
# 输出： xx大楼
```

**修改**
```py
building['height'] = 72
```

#### 1.2.3 元组

元组作为 Python 中有序不可更改的集合，虽然说用元组能实现的功能用列表也能做到，但是元组的特性，也给开发者带来不少便利。

上述变量赋值操作可以用元组赋值的方法，进行多个变量的赋值

```py
# 实际上就是创建了一个元组，包含两个元素，一个是1，一个是2，然后分别赋值给a，b
a, b = 1, 2

# 相当于是
c = 1, 2
a, b = c
```

```py

# 也可以两个以上的
t1 = 2, 4, 6
# 也可以是不同类型的
t2 = 'hi', True, 6, 7.5
```

{{< hint info >}}
**Tip**  
也就是说，在 Python 中，两个变量或者值，直接用逗号隔开，不用中括号表示的就是元组
{{< /hint >}}

因为元组是有序的，所以可以通过索引符号取值，但是不能为元组的某一项进行再一次赋值，不能删除，是只读的

**取值**
```py
# 元组也可以通过索引符号取值
c[1]
# 输出： 2
```

{{< hint warning >}}
**注意**  
由于只能从元组里取值，故以下修改，删除是错误的写法，也不能在元组声明后新增新的元组元素
```py
# 但是不能赋值，如下表达式就是错误的，因为元组不可更改
c[1] = 3
del c[0]
```
{{< /hint >}}

#### 1.2.4 集合

集合是无序不重复序列，不可以通过索引取值，如下创建一个集合

```py
s1 = {1, 2, 4, 6, 3}
```

{{< hint info >}}
**Tip**  
也就是说，元组包上大括号就是集合了
{{< /hint >}}

由于集合 Set 中不能有重复的元素，所以在创建集合时的重复元素会被去掉，并且重新排序

```py
s2 = {2, 3, 1, 4, 3, 1, 6, 2}
# 输出： {1, 2, 3, 4, 6}
```

**新增**
```py
s2.add(9)
s2.add(3)
# 输出：{1, 2, 3, 4, 6, 9}
```

**删除**
```py
s2.remove(3)
# 输出: {1, 2, 4, 6, 9}
```

{{< hint info >}}
**Tip**  
集合创建后元素不能被被修改
{{< /hint >}}

{{< hint info >}}
**Tip**  
集合没有可以直接查询集合内元素的方法，但是可以通过遍历获取，参照[循环遍历](#32-循环遍历-for)
{{< /hint >}}

{{< details "扩展内容" >}}
#### 集合的运算

```py
a = {'a', 'b', 3, 'c', 'd', 5}
b = {'1', 'a', 4, 'd', 2}

a - b  # a 和 b 的差集
# 输出：{'c', 3, 'b', 5}

a | b  # a 和 b 的并集
# 输出：{'c', 2, 3, 4, 5, 'b', 'd', '1', 'a'}

a & b  # a 和 b 的交集
# 输出：{'a', 'd'}

a ^ b  # a 和 b 中不同时存在的元素
# 输出：{2, 3, 4, 5, 'c', 'b', '1'}
```
{{< /details >}}
