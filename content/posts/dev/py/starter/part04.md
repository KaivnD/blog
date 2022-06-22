---
title: "Python起步 04"
date: 2022-06-15T00:58:24+08:00
tags: ["python"]
summary: python快速起步
---

## Part 4 对象与模块

使用抽象思维建立一些模块可以使得程序变得更容易阅读，充分分离各个功能组团，可以使代码复用率提升。
有效的模块封装可以让自己写的功能更容易在多个项目中使用，同时也可以在社区获取别的大神所创造的功能，
丰富的功能模块，使得 Python 更强大。

### 4.1 面对对象编程

**面对对象编程**是一种**抽象**思维，通常和**面对过程编程**做比较，本文从开头到此都是面对过程的编程，
直接从过程入手，通过流程控制变量的数据流动，完成指定功能。这两种编程思维没有优劣之分，相辅相成。
**面对对象编程**的思想用于宏观思考，组织架构，**面对过程编程**的思想即可用解决宏观架构下的微观问题，
又可以辅助建立宏观框架。

#### 4.1.1 模版

**抽象**是一个核心思维，让我们回到先前提到的[字典的例子](#122-字典)，这就是一个典型的抽象思维，
我们在这里定义了一个建筑物的数据结构，他有名称，类型，高度，楼层等信息，这就可以看作是一个对象，
而名称，类型，高度，楼层等信息是这个对象的基本属性。

```py
building = {
    'name': 'xx大楼',
    'type': '公建',
    'height': 99,
    'floors': [
        {
            'name': 'L0',
            'height': 3
        },
        {
            'name': 'L1',
            'height': 4.5
        }
    ]
}
```

如果我们能确定这些基本属性的值，那么也就确定了一个建筑物：

1. 这些基本属性的集合可以称为创建一个建筑物的模版，也可以叫做一个**类（Class）**
2. 通过这些属性的值得到的建筑物叫做**对象**，也可以叫做**实例**
3. 把值代进去创建一个建筑物的过程叫**实例化**
4. 每一个属性都是这个类的**成员**，这里的名称，类型，高度，楼层都是变量，所以又叫做**成员变量**或者**成员属性**

下面我们用 Python 把这个模版表示出来：

```py
# 声明一个名为Building的class
class Building:
    '''
    定义构造函数，实例化的时候将会被调用，函数也是类的成员。
    第一个参数为形参，必须要带，只有带上这个形参，才是成员函数，名字无所谓，通常写成self，表示自己
    第二个参数为名称，是自己加的，想加什么参数就可以加什么参数，加的参数需要在实例化的时候填值，表示这栋楼的名称
    第三个参数类型，表示建筑类型
    '''
    def __init__(self, name, buildingt_type):
        # 构造函数内需要对属性进行初始化，可以从构造参数传递，也可以直接赋值
        self.name = name
        #通过.操作符，可以用访问self的成员
        self.type = buildingt_type
        self.floors = []

    '''
    定义一个增加楼层的方法（函数），class的成员函数一般被称为方法，
    这个方法，用来增加一个楼层，将会改变floors成员的值
    '''
    def addFloor(self, name, height):
        self.floors.append((name, height))

    '''
    定义一个方法来获取这栋楼的高度，因为楼的高度是随着楼层的改变而改变，
    所以这个成员的值是与别的变量有相关性的。所以使用一个函数来计算，
    函数头上有一个修饰符@property用来说明这个height成员是一个计算属性，
    而不是一个方法。
    '''
    @property
    def height(self):
        # 定义一个变量储存高度信息
        height = 0

        # 遍历每一个楼层
        for floor in self.floors:
            # 把楼层高度通过加法赋值累加到height变量
            height += floor[1]

        # 返回所有楼层高度之和
        return height
```

#### 4.1.2 对象

上述模版仅仅只是规定了`Building`这个对象长什么样，相当于是一个**模具**，需要拿到生产线加工才能得到**对象**。

1. 案例 A

```py
# 起一个变量名为building，构造一个名称为xx大楼，类型为公建的Building的实例
building = Building("xx大楼", "公建")
# 调用building的addFloor方法为Building增加一层名为L0，高度为4.5的楼层
building.addFloor("L0", 4.5)
# 调用building的addFloor方法为Building增加一层名为L1，高度为3的楼层
building.addFloor("L1", 3)

# 获取building的高度
print(building.height)
# 输出：4.5
```

2. 案例 B

```py
# 起一个名为buildings的空列表
buildings = []

for i in range(10):
    name = "#" + str(i + 1)
    building = Building(name, "多层")

    # 第二层循环用于给每一个building增加一定数量的楼层
    for j in range(6):
        floor = "L" + str(j)
        building.addFloor(floor, 3)

    buildings.append(building)

# 这样以来，就可以有一系列的建筑物储存在一个列表当中
for building in buildings:
    print("[{}楼] 有{}m高，一共有{}层".format(building.name, building.height, len(building.floors)))

"""
输出：
[#1楼] 有18m高，一共有6层
[#2楼] 有18m高，一共有6层
[#3楼] 有18m高，一共有6层
[#4楼] 有18m高，一共有6层
[#5楼] 有18m高，一共有6层
[#6楼] 有18m高，一共有6层
[#7楼] 有18m高，一共有6层
[#8楼] 有18m高，一共有6层
[#9楼] 有18m高，一共有6层
[#10楼] 有18m高，一共有6层
"""
```

{{< hint info >}}
**Tips**  
**字符串拼接** `"{}，{}".format(var1, var2)` 这种格式是常用的字符串拼接形式，可以方便灵活完成拼接，保证字符串模板和变量分离。 参照[Advanced String Formatting](https://peps.python.org/pep-3101/)
{{< /hint >}}

#### 4.1.3 Building class

经过前面对 Building 对象的描述，可以发现还是有些缺陷，比如说缺少对楼层的抽象，缺少楼层具体形状。
这一小节，重新来对 Building 对象进行完善。
声明一个 Floor 的 class，用来做建筑的楼层的模板

```py
# 声明一个楼层的模板，包含三个属性，名称，高度，和边界，用于描述楼层的形状
class Floor:
    def __init__(self, name, height, bound):
        self.name = name
        self.height = height
        self.bound = bound

# 摘自上文的Building 去掉了一些暂时用不到的东西
class Building:
    def __init__(self, name, buildingt_type):
        self.name = name
        self.type = buildingt_type
        self.floors = []

    """
    使用跟简洁的写法来求总楼高，sum函数可以对数组进行求和。
    这个只包含楼层高度的数组的写法，详见3.2.2 遍历列表的扩展内容。
    """
    @property
    def height(self):
        return sum([floor.height for floor in self.floors])
```

上面的程序缺失了部分内容，就是`Floor`的`bound`，应该是个多边形，来描述楼层的具体形状。
一个多边形，由多个点组成，这里我们抽象一个多边形`Polyline2d`的模板，用元组来储存多边形的各个点，
我们甚至还可以写一个函数，求解多边形的面积。

```py
class Polyline2d:
    """
    构造一个多边形，包含一个pts属性，用于储存多边形的各个顶点
    """
    def __init__(self):
        self.pts = []

    """
    add 方法传入两个值，一个x坐标，一个y坐标
    """
    def add(self, x, y):
        # 用元组的形式，把x坐标和y坐标打包成一个数据，储存到self.pts属性内
        self.pts.append((x, y))

    """
    鞋带公式求解已知多边形所有顶点的情况的多边形面积，详见https://www.101computing.net/the-shoelace-algorithm/
    """
    @property
    def area(self):
        s1 = 0
        s2 = 0
        cnt = len(self.pts)

        for i in range(cnt - 1):
            s1 += self.pts[i][0] * self.pts[i+1][1]
            s2 += self.pts[i][1] * self.pts[i+1][0]

        s1 += self.pts[cnt-1][0] * self.pts[0][1]
        s2 += self.pts[0][0] * self.pts[cnt-1][1]

        return abs(s1 - s2) / 2
```

这样以来，我们除了直接得到建筑物的总高度以外，还可以得到总面积。
为 Building 增加计算总面积的属性

```py
class Building:
    #...省略...

    @property
    def area(self):
        return sum([floor.bound.area for floor in self.floors])
```

我们可以为 class 提供转换为字符串的方法，这样在 print 的时候以一个固定格式显示变量，看起来方便一些。

```py
class Floor:
    #...省略...
    def __str__(self) -> str:
        return "{}       {}     {:.2f}".format(self.name, self.height, self.bound.area)

class Building:
    #...省略...
    def __str__(self) -> str:
        levels = "\n-------------------------\n".join([str(floor) for floor in self.floors])
        return """-------------------------
建筑名称：{}
建筑类型：{}
建筑高度：{} m
建筑层数：{} 层
建筑面积：{:.2f} m²

楼层    层高    面积
-------------------------
{}
-------------------------""".format(self.name, self.type, self.height, len(self.floors), self.area, levels)
```

{{< hint info >}}
**Tips**  
**跨行字符串** 三个引号可表示跨行字符串。
{{< /hint >}}

这样一来一个粗糙的`Building`就实现了，在这个基础之上可以深入设计更多的功能，就目前的成果，试一下效果：

```py
# 假设我们有这一组顶点，作为楼层的形状多边形的顶点。
pts = [
    (8.1, -9.483704),
    (8.1, -11.378272),
    (-8.1, -11.378272),
    (-8.1, -9.483704),
    (-16.2, -9.483704),
    (-16.2, 1.67),
    (-6.8, 1.67),
    (-6.8, 3.4),
    (-1.5, 3.4),
    (-1.5, -1.8),
    (1.5, -1.8),
    (1.5, 3.4),
    (6.8, 3.4),
    (6.8, 1.67),
    (16.2, 1.67),
    (16.2, -9.483704),
    (8.1, -9.483704),
    (8.1, -9.483704)
]

# 创建一个多边形
pl = Polyline2d()

# 把顶点数组的所有顶点添加到多边形中
for x, y in pts:
    pl.add(x, y)

# 创建一个建筑实例
building = Building("xx大楼", "公建")

# 楼层信息
floor_heights = [4.5, 3, 3, 4.8, 3, 3]

# 一次性创建6个一样的楼层
for i, height in enumerate(floor_heights):
    floor = Floor("F{}".format(i + 1), height, pl)
    building.floors.append(floor)

print(building)

"""
输出：
(base) kaivnd@KaivnDdeMacBook-Pro blog % python test.py
-------------------------
建筑名称：xx大楼
建筑类型：公建
建筑高度：21.3 m
建筑层数：6 层
建筑面积：2400.00 m²

楼层    层高    面积
-------------------------
F1       4.5     400.00
-------------------------
F2       3       400.00
-------------------------
F3       3       400.00
-------------------------
F4       4.8     400.00
-------------------------
F5       3       400.00
-------------------------
F6       3       400.00
-------------------------
"""
```

{{< details "合在一起" >}}

```py
class Polyline2d:
    """
    构造一个多边形，包含一个pts属性，用于储存多边形的各个顶点
    """
    def __init__(self):
        self.pts = []

    """
    add 方法传入两个值，一个x坐标，一个y坐标
    """
    def add(self, x, y):
        # 用元组的形式，把x坐标和y坐标打包成一个数据，储存到self.pts属性内
        self.pts.append((x, y))

    """
    鞋带公式求解已知多边形所有顶点的情况的多边形面积，详见https://www.101computing.net/the-shoelace-algorithm/
    """
    @property
    def area(self):
        s1 = 0
        s2 = 0
        cnt = len(self.pts)

        for i in range(cnt - 1):
            s1 += self.pts[i][0] * self.pts[i+1][1]
            s2 += self.pts[i][1] * self.pts[i+1][0]

        s1 += self.pts[cnt-1][0] * self.pts[0][1]
        s2 += self.pts[0][0] * self.pts[cnt-1][1]

        return abs(s1 - s2) / 2

# 声明一个楼层的模板，包含三个属性，名称，高度，和边界，用于描述楼层的形状
class Floor:
    def __init__(self, name, height, bound):
        self.name = name
        self.height = height
        self.bound = bound

    def __str__(self) -> str:
        return "{}       {}     {:.2f}".format(self.name, self.height, self.bound.area)

# 摘自上文的Building 去掉了一些暂时用不到的东西
class Building:
    def __init__(self, name, buildingt_type):
        self.name = name
        self.type = buildingt_type
        self.floors = []

    def __str__(self) -> str:
        levels = "\n-------------------------\n".join([str(floor) for floor in self.floors])
        return """-------------------------
建筑名称：{}
建筑类型：{}
建筑高度：{} m
建筑层数：{} 层
建筑面积：{:.2f} m²

楼层    层高    面积
-------------------------
{}
-------------------------""".format(self.name, self.type, self.height, len(self.floors), self.area, levels)

    """
    使用跟简洁的写法来求总楼高，sum函数可以对数组进行求和。
    这个只包含楼层高度的数组的写法，详见3.2.2 遍历列表的扩展内容。
    """
    @property
    def height(self):
        return sum([floor.height for floor in self.floors])

    @property
    def area(self):
        return sum([floor.bound.area for floor in self.floors])


# 假设我们有这一组顶点，作为楼层的形状多边形的顶点。
pts = [
    (8.1, -9.483704),
    (8.1, -11.378272),
    (-8.1, -11.378272),
    (-8.1, -9.483704),
    (-16.2, -9.483704),
    (-16.2, 1.67),
    (-6.8, 1.67),
    (-6.8, 3.4),
    (-1.5, 3.4),
    (-1.5, -1.8),
    (1.5, -1.8),
    (1.5, 3.4),
    (6.8, 3.4),
    (6.8, 1.67),
    (16.2, 1.67),
    (16.2, -9.483704),
    (8.1, -9.483704),
    (8.1, -9.483704)
]

# 创建一个多边形
pl = Polyline2d()

# 把顶点数组的所有顶点添加到多边形中
for x, y in pts:
    pl.add(x, y)

# 创建一个建筑实例
building = Building("xx大楼", "公建")

# 楼层信息
floor_heights = [4.5, 3, 3, 4.8, 3, 3]

# 一次性创建6个一样的楼层
for i, height in enumerate(floor_heights):
    floor = Floor("F{}".format(i + 1), height, pl)
    building.floors.append(floor)

print(building)
```

{{< /details >}}

### 4.2 模块

用一句话讲明白模块就是，把代码组织在相应文件，相应的包里，这样一来，
可以做到问题细分，任务细分。在改某个功能的时候，专注于这个功能，
没有必要把一个程序文件写老长，维护起来很麻烦。可以把一些已经固定用途的功能，
放到一个包内，提供给团队，或者上传至网络上与大家分享，这样一来，即做到自己省心，
又做到大家省心，多棒！

#### 4.2.1 代码组织

上一节内容我们最后把所有的代码放到了一个文件里，在这一届可以做一个简单的代码组织练习。

第一个文件用来保存几何图形的代码

```py
"""
@file geometry.py
"""
class Polyline2d:
    """
    构造一个多边形，包含一个pts属性，用于储存多边形的各个顶点
    """
    def __init__(self):
        self.pts = []

    """
    add 方法传入两个值，一个x坐标，一个y坐标
    """
    def add(self, x, y):
        # 用元组的形式，把x坐标和y坐标打包成一个数据，储存到self.pts属性内
        self.pts.append((x, y))

    """
    鞋带公式求解已知多边形所有顶点的情况的多边形面积，详见https://www.101computing.net/the-shoelace-algorithm/
    """
    @property
    def area(self):
        s1 = 0
        s2 = 0
        cnt = len(self.pts)

        for i in range(cnt - 1):
            s1 += self.pts[i][0] * self.pts[i+1][1]
            s2 += self.pts[i][1] * self.pts[i+1][0]

        s1 += self.pts[cnt-1][0] * self.pts[0][1]
        s2 += self.pts[0][0] * self.pts[cnt-1][1]

        return abs(s1 - s2) / 2
```

第二个文件用来保存建筑物相关的代码

```py
"""
@file building.py
"""
class Floor:
    def __init__(self, name, height, bound):
        self.name = name
        self.height = height
        self.bound = bound

    def __str__(self) -> str:
        return "{}       {}     {:.2f}".format(self.name, self.height, self.bound.area)


class Building:
    def __init__(self, name, buildingt_type):
        self.name = name
        self.type = buildingt_type
        self.floors = []

    def __str__(self) -> str:
        levels = "\n-------------------------\n".join([str(floor) for floor in self.floors])
        return """-------------------------
建筑名称：{}
建筑类型：{}
建筑高度：{} m
建筑层数：{} 层
建筑面积：{:.2f} m²

楼层    层高    面积
-------------------------
{}
-------------------------""".format(self.name, self.type, self.height, len(self.floors), self.area, levels)

    """
    使用跟简洁的写法来求总楼高，sum函数可以对数组进行求和。
    这个只包含楼层高度的数组的写法，详见3.2.2 遍历列表的扩展内容。
    """
    @property
    def height(self):
        return sum([floor.height for floor in self.floors])

    @property
    def area(self):
        return sum([floor.bound.area for floor in self.floors])
```

第三个文件用来保存主程序代码

```py
"""
@file main.py
"""
# 假设我们有这一组顶点，作为楼层的形状多边形的顶点。
pts = [
    (8.1, -9.483704),
    (8.1, -11.378272),
    (-8.1, -11.378272),
    (-8.1, -9.483704),
    (-16.2, -9.483704),
    (-16.2, 1.67),
    (-6.8, 1.67),
    (-6.8, 3.4),
    (-1.5, 3.4),
    (-1.5, -1.8),
    (1.5, -1.8),
    (1.5, 3.4),
    (6.8, 3.4),
    (6.8, 1.67),
    (16.2, 1.67),
    (16.2, -9.483704),
    (8.1, -9.483704),
    (8.1, -9.483704)
]

# 创建一个多边形
pl = Polyline2d()

# 把顶点数组的所有顶点添加到多边形中
for x, y in pts:
    pl.add(x, y)

# 创建一个建筑实例
building = Building("xx大楼", "公建")

# 楼层信息
floor_heights = [4.5, 3, 3, 4.8, 3, 3]

# 一次性创建6个一样的楼层
for i, height in enumerate(floor_heights):
    floor = Floor("F{}".format(i + 1), height, pl)
    building.floors.append(floor)

print(building)
```

把这三个文件放到同一个文件夹里，尝试执行`python main.py`，得到以下错误：

```bash
(base) kaivnd@KaivnDdeMacBook-Pro blog % python test.py
Traceback (most recent call last):
  File "main.py", line 27, in <module>
    pl = Polyline2d()
NameError: name 'Polyline2d' is not defined
```

很显然，这个`main.py`文件内使用的 class `Polyline2d` 以及别的 class，都被我们移动到另一个文件了。

#### 4.2.2 import 语句

通过`import`语句把丢失的`class` “找”回来，`import`一般写在文件开头，
有以下几种常用写法：

- `import geometry` 这样是把整个 geometry 文件当成一个模块导入进来，使用方法如下

```py
import geometry

# ...省略...

# 创建一个多边形
pl = geometry.Polyline2d()
```

- `import geometry as g` 这样和上面一样，只不过是给`geometry`起了个新名字`g`

```py
import geometry as g

# ...省略...

# 创建一个多边形
pl = g.Polyline2d()
```

- `from geometry import Polyline2d` 这样的导入是直接导入需要用的类型、函数、或者变量

```py
from geometry import Polyline2d

# ...省略...

# 创建一个多边形
pl = Polyline2d()
```

我们将缺失的`class`的`import`补全，`main.py`文件的开头就变成这样子

```py
"""
@file main.py
"""
from geometry import Polyline2d
from building import Floor, Building
# ...省略...
```

这样一来，一个简单的代码组织工作就做完了。

```bash
├─example
│  ├─building.py # 建筑信息模块
│  ├─geometry.py # 几何图形模块
│  └─main.py # 主程序
```

{{< button href="/archive/py-starter-example.zip" >}}下载代码{{< /button >}}