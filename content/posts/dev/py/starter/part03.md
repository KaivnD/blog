---
title: "Python起步 03"
date: 2022-06-15T00:58:24+08:00
tags: ["python"]
summary: python快速起步
---


## Part 3 流程与控制

我们写的大部分代码都是在描述流程，一个程序从开始到结束，变量在其中受流程语句的控制，有秩序的流动。
可以说，流程是一个程序的血管，而变量就是流淌在血液中的细胞。

### 3.1 条件判断 if...else...

条件判断可以根据变量的不同值，产生不同的运行路径，如上求一元二次方程的根，我们知道当**判别式**{{< katex >}}\Delta = b^2 - 4ac{{< /katex >}}大于等于`0`时，方程在实数范围内有解，所以，在这个函数内，我们需要区分**判别式**的不同情况，为函数返回不同的值，这就叫做条件判断，让我们来完善此函数

```py
def f(a, b, c):
····delta = b**2 - 4 * a * c # 判别式
····if delta > 0:
········# 此时方程有两个不相等的实根
········sqrt_delta = delta ** 0.5 # 判别式的平方根
········return (-b + sqrt_delta) / 2*a, (-b - sqrt_delta) / 2*a # 以元组形式返回两个根
····elif delta == 0:
········# 此时方程有两个相等的实根
········sqrt_delta = delta ** 0.5 # 判别式的平方根
········return (-b + sqrt_delta) / 2*a
····else:
········# 此时该方程在实数范围内无解，返回一个None代表无解
········return None
```

{{< hint info >}}
**Tip**
注意看每一行的缩进，这一次有两种缩进，一个是一个单位（4 个），另一个是两个单位（8 个）。由不同单位的缩进代表了，每一行代码属于的代码块，特别注意**冒号**的使用
{{< /hint >}}

{{< hint info >}}
**Tip**
此函数中`return` 语句代表函数已经执行完成，后续流程不在执行。
{{< /hint >}}

{{< details "扩展内容" >}}

#### 三元表达式

三元表达式可在简短的语句内完成有条件的变量声明，Python 中没有和其他语言一样的三元表达式，但是有代替的写法：
**形式：** `A if C else B`，其中，当`C`成立（值为True）时，值为`A`，否则值为`B`，`A, B, C` 就叫做三元

```py
# 变量number
number = 12
# number是奇数还是偶数，使用类似三元表达式来赋值
number_type = '偶数' if number % 2 == 0 else '奇数'
```

上述表达式中，`'偶数' if number % 2 == 0 else '奇数'`就是三元表达式，读成一句话就是，这个表达式的值是偶数如果 number 除以 2 的余数等于 0 的话，否则就是奇数。

{{< / details >}}

### 3.2 循环遍历 for...

循环是一个很常用的流程，很多时候都需要重复的执行某些操作，比如：

#### 3.2.1 遍历等差数列

> [`range(...)`](https://docs.python.org/3/library/functions.html#func-range)函数
> Python 的一个内建功能，提供生成等差数列的功能

**函数签名**

> **range(stop)** 生成首项为 0，末项为 stop - 1 的等差数列，公差为 1

> **range(start, stop[, step])** 生成首项为 start，末项为 stop - 1 的等差数列，公差 step 为可选项，不填就是 1

> [`print(...)`](https://docs.python.org/3/library/functions.html#func-print)函数
> Python 的一个内建功能，可把变量的值输出到控制台查看

```py
for i in range(5):
    print(i)
"""
输出：
0
1
2
3
4
"""

for i in range(0, 4):
    print(i)
"""
输出：
0
1
2
3
"""

for i in range(0, 10, 2):
    print(i)
"""
输出：
0
2
4
6
8
"""
```

#### 3.2.2 遍历列表

遍历列表就是起一个变量，代表列表中的每一个元素，对元素进行操作。
如下程序，将 a 列表里不是偶数的放到 b 列表中

```py
a = [3, 5, 2, 6, 7, 9, 11, 23]
b = []
for item in a:
    # % 是求余数运算符，这里是把列表里的每一个数字都除以2，然后得到余数，若除以2的余数为零，代表能除尽，就是偶数
    if item % 2 != 0:
        b.append(item)

print(b)
# 输出：[3, 5, 7, 9, 11, 23]
```

其中，`for`代码块内的变量`item`就被看成是列表中的每一个元素

{{< details "扩展内容" >}}

#### 列表解析

上述程序可用[列表解析](https://peps.python.org/pep-0202/)更简短的表达

```py
a = [3, 5, 2, 6, 7, 9, 11, 23]
b = [item for item in a if item % 2 != 0]
```

{{< /details >}}

有时候我们需要，同时遍历索引和值可以这么写：

```py
for i in range(len(a)):
    val = a[i]
    print(i, val)
'''
输出：
0 3
1 5
2 2
3 6
4 7
5 9
6 11
7 23
'''
```

> [`len(...)`](https://docs.python.org/3/library/functions.html#enumerate)函数
> Python 的一个内建功能，在这里是列表长度

{{< details "扩展内容" >}}

#### 另一种写法

```py
for i, val in enumerate(a):
    print(i, val)
# 和第一种写法效果一样
```

> [`enumerate(...)`](https://docs.python.org/3/library/functions.html#enumerate)函数
> Python 的一个内建功能，获取可迭代对象的迭代器，对于这里的列表而言，迭代器就是一个索引和值组成的元组
> {{< /details >}}

#### 3.2.3 遍历字典

遍历字典和遍历列表类似，不过因为字典是多个键值对，情况稍微不太一样。

1. 如果按照列表的遍历来写的话，将只能遍历键

```py
building = {
    'name': 'xx大楼',
    'type': '公建',
    'height': 99
}

for item in building:
    # item 相当一是字典的每一个key
    print(item)
    # 可以通过索引操作符取值
    val = building[item]
"""
输出：
name
type
height
"""

# 相当于
for item in building.keys():
    print(item)
```

2. 要遍历值的话需要指明：

```py
for item in building.values():
    print(item)
"""
输出：
xx大楼
公建
99
"""
```

3. 遍历键值对

```py
for item in building.items():
    print(item)
"""
输出：
('name', 'xx大楼')
('type', '公建')
('height', 99)
"""
```

可以看出，这个 item 是个元组，那我们可以使用[元组赋值](#123-元组)的方式起两个变量：

```py
for key, val in building.items():
    print(key, val)
"""
输出：
name xx大楼
type 公建
height 99
"""
```

#### 3.2.4 跳过 continue

回到遍历数组那个例子，这里描述的是当`item`不是偶数的时候把 item 放到列表 b 里，
反过来就是，如果`item`是奇数，就**跳过**这次循环，**继续**下次循环，这就是`continue`，如下示例：

```py
a = [3, 5, 2, 6, 7, 9, 11, 23]
b = []
for item in a:
····if item % 2 == 0: continue
····b.append(item)

print(b)
# 输出：[3, 5, 7, 9, 11, 23]
```

所以`continue`继续的意思是继续**下一次循环**，一般我这么写是为了节省代码块，读起来更通顺。
特别是需要很多条件判断罗列、流程比较复杂的时候，这么写会更简短些。

#### 3.2.5 跳出 break

跳出和跳过是完全不一样的作用，跳出表示终止一层循环，比方说：
在一串字符中找到第一个字母`n`是字符串的第几个字母，如果找到了，后续的循环都是没有必要的，可以直接 break

```py
words = 'Hi noah!'

index = -1

# 字符串可看作不可变有序的一种序列
for i in range(len(words)):
    letter = words[i]
    if letter == 'n':
        index = i
        break

if index > 0:
    # 拼接字符串，见扩展内容
    print('字母n在 ' + words + ' 中是第' + str(index) + '个字母')
else:
    print('没有在该字符串中找到字母n')

# 输出： 字母n在 Hi noah! 中是第3个字母
```

> [`str(...)`](https://docs.python.org/3/library/functions.html#func-str)函数
> Python 的一个内建功能，将变量转为字符串类型

{{< details "扩展内容" >}}

#### 字符串拼接

由于字符串可看作不可变有序的一种序列，所以可以使用`+`进行连接，单这种方式比较笨拙，容易阅读。
可用[Advanced String Formatting](https://peps.python.org/pep-3101/)，简化此过程。

```py
message = "字母n在 {} 中是第 {} 个字母".format(words, index) if index > 0 else '没有在该字符串中找到字母n'
print(message)
```

和上面的写法效果完全一样，推荐使用这种方法做字符串拼接
{{< /details >}}

### 3.3 条件循环 while...

`for` 循环一般用来做可遍历对象的遍历，当可遍历对象遍历完之后，循环就结束了。
`while` 循环是根据条件来循环，如果条件成立，将始终一遍一遍的循环下去，除非外力干扰。如：

```py
while True:
    print('never ends loop!')
```

`while`就是当条件满足的时候，执行 while 代码块，这类循环通常在当你需要在某个条件不满足时才停止的循环，或者不确定循环多少次的循环，比如：

```py

i = 0

while i < 6:
    # 加法赋值运算，见扩展内容
    i += 1
    print(i)
'''
输出：
1
2
3
4
5
6
'''
```

也可以写成

```py
i = 0
while True:
    i  += 1
    if i > 6:
        break
    print(i)
```

{{< details "扩展内容" >}}

#### Python 赋值运算符

```py
i = 0
i += 1 # 相当于 i = i + 1，也就是说，旧的i值加1赋值给新的i
```

| 符号  |       名称       |
| :---: | :--------------: |
|   =   |    赋值运算符    |
|  +=   |  加法赋值运算符  |
|  -=   |  减法赋值运算符  |
|  \*=  |  乘法赋值运算符  |
|  /=   |  除法赋值运算符  |
|  %=   |  取模赋值运算符  |
| \*\*= |   幂赋值运算符   |
|  //=  | 取整除赋值运算符 |

{{< /details >}}

### 3.4 异常捕捉 try...except...

要说异常的捕捉得先说什么是异常，先倒回到求实数根函数，对于这个函数的设定，
是否有实根是可以通过判别式判断的，我们可以对这个函数进行设定，若无实根，认为是一种异常。如下所示：

```py
def f(a, b, c):
    delta = b**2 - 4 * a * c # 判别式
    if delta > 0:
        # 此时方程有两个不相等的实根
        sqrt_delta = delta ** 0.5 # 判别式的平方根
        return (-b + sqrt_delta) / 2*a, (-b - sqrt_delta) / 2*a # 以元组形式返回两个根
    elif delta == 0:
        # 此时方程有两个相等的实根
        sqrt_delta = delta ** 0.5 # 判别式的平方根
        return (-b + sqrt_delta) / 2*a
    else:
        # 设定，若无实根，认为是一种异常，那就不返回值，抛出一个异常
        raise Exception('输入条件无实根')
```

那么这个函数在被调用的时候，如果没有实根，则会直接抛出异常，退出整个程序。

```py
f(1, 2, 3)
'''
输出：
(base) kaivnd@KaivnDdeMacBook-Pro blog % python test.py
Traceback (most recent call last):
  File "test.py", line 15, in <module>
    f(1, 2, 3)
  File "test.py", line 13, in f
    raise Exception('输入条件无实根')
Exception: 输入条件无实根
'''
```

这个界面将会是大家日后经常会看到的，从最后一行的输出来看，正式我们在函数的末尾所抛出的异常。
`raise` 关键字用于抛出异常，`Exception`是一种异常。那么如何在明知函数会有异常的时候保证程序正常工作呢？
那就是要**捕捉异常**了

使用`try...except...`语句进行异常捕捉

```py
try:
    f(1, 2, 3)
except:
    print('这个f函数有异常，不太行，要处理一下这个函数遇到异常的情况。')

'''
输出：
(base) kaivnd@KaivnDdeMacBook-Pro blog % python test.py
这个f函数有异常，不太行，要处理一下这个函数遇到异常的情况。
'''
```

上述代码块仅是在找到异常的时候提个醒。通常，异常捕捉需要结合上下文来看，
比方说这里`try`内的流程是为了取到某个值，然后赋值给这个`try`代码块外的变量，
那么当遇到异常的时候，应该考虑使用另一种方法来，来计算这个值。

另，这个`except`部分并没有变量表示我们在 f 函数里抛出来的错误信息，我们可以这样来获取：

```py
try:
    f(1, 2, 3)
except Exception as err:
    print(err)
'''
输出：
(base) kaivnd@KaivnDdeMacBook-Pro blog % python test.py
输入条件无实根
'''
```

{{< details "扩展内容" >}}

#### 多种异常的捕捉

我们知道，二元一次方程，的第一元，二次方的常数部分如果是 0 的话，就相当于是一元一次方程了。
所以在这个函数的设定下，我们也可以认为 a 为 0 是一种异常

```py
def f(a, b, c):
    if a == 0:
        raise ValueError("a 不能为0")
    delta = b**2 - 4 * a * c # 判别式
    if delta > 0:
        # 此时方程有两个不相等的实根
        sqrt_delta = delta ** 0.5 # 判别式的平方根
        return (-b + sqrt_delta) / 2*a, (-b - sqrt_delta) / 2*a # 以元组形式返回两个根
    elif delta == 0:
        # 此时方程有两个相等的实根
        sqrt_delta = delta ** 0.5 # 判别式的平方根
        return (-b + sqrt_delta) / 2*a
    else:
        # 设定，若无实根，认为是一种异常，那就不返回值，抛出一个异常
        raise Exception('输入条件无实根')
```

这样的话在捕捉异常的时候就可以分清楚是哪种异常了。

```py
try:
    f(1, 2, 3)
except ValueError as ve:
    # 这个情况就是a为0了
    print(ve)
except Exception as e:
    # 这种情况就是没有实根
    print(e)
```

#### 完整形态

`try...except...`的完整形态其实是`try...except...else...finally...`

```py
try:
    f(1, 2, 3)
except:
    print('有异常')
else:
    print('没有异常')
finally:
    print('管你有没有异常都要执行')
```

#### with 语句

`with`语句使用来缩短`try...finally...`的语法。举个常用的例子，从硬盘读取要一个文件，通常需要这么写：

1. 不捕捉异常

```py
file = open("/path/to/file.txt")
data = file.read()
file.close()
```

2. 捕捉异常

```py
file = open("/path/to/file.txt")
try:
    data = file.read()
finally:
    file.close()
```

也就是说，我们通过 python 从硬盘里读取一个文件完毕后时需要调用 close 取关闭这个文件，告诉系统你对这个文件已经使用完了。
假设，在读取的过程遇到异常，如果使用第一种写法在执行到`file.read()`到时候直接抛出异常，那就以为着整个程序瞬间结束，
`file.close()`压根儿没有机会执行，所以你这时候使用我们刚学的知识，可以使用`try...finally...`管你遇不遇到异常，
都要执行这个`file.close()`，也就是说，就算在读取文件的时候遇到异常，文件也可以被正确关闭。但是这么写，太长了，太复杂，
所以就有了`with`，如下，使用`with`来写：

```py
with open("/path/to/file.txt") as f:
    content = f.read()
```

> 简短且安全的完成文件读取
> {{< /details >}}
