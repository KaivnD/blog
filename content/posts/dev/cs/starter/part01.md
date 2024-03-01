---
title: "C# 01"
date: 2024-02-28T22:16:13+08:00
# bookComments: false
# bookSearchExclude: false
---
# 变量·集合

## 类型系统

### 内置类型（原始类型）
- 值类型

> *KN01* 展开说明所有值类型

```cs
bool v0 = true; // 布尔，只有两个值，true或者false
int v1 = 32; // 整数
double v2 = 3.14; // 浮点
char v3 = 'n'; // 字符
```

- 引用类型
```cs
object v4 = null; // 源头
string v5 = "ncf"; // 字符串
```

- 缺少/没有类型
标记为
```cs
void
```

### 定义类型

通常指使用 `struct`、`class`、`interface`、`enum` 和 `record` 结合原始类型来 构造出新的 **数据结构** 类型
由 `struct` 和 `enum` 构造出来的是值类型，比如说，

```cs
struct Point3d 
{
    double x, y, z;
}

Point3d pt;
```

由`class`构造的是引用类型 

```cs
class Curve
{
}
Curve crv = new();
```

> *KN02* 类型成员详解  
> *KN03* 变量分类： 静态变量、实例变量、数组元素、值参数、引用参数、输出参数和局部变量  
> *KN04* `struct`和`class`的区别，以及适用情况

`interface` 是一个特殊的存在，他虽然能定义一种类型，但是它并不负责具体的实现，是一种约定，是一种规则，叫做**接口**。

```cs
interface IJsonData
{
    string ToJson();
}
```

一套**源代码**包含很多个这样的定义，合在一起可以**编译**成一个库，编译好的库可以被别的源代码引用，方便组织功能或者扩展功能。

- `.Net Framework` 是Microsoft写到操作系统里的库，在Windows环境下，我们可以引用它定义好的库，从而调用操作系统层面的功能。
    - System.IO
    - System

[.NET API 浏览器](https://learn.microsoft.com/zh-cn/dotnet/api/?view=netframework-4.8)

- `RhinoCommon` 是McNeel提供的三维空间几何运算功能的库，只能在Rhino软件环境下调用。
    - Rhino.Geometry
    - Rhino.DocObjects

[RhinoCommon API 浏览器](https://mcneel.github.io/rhinocommon-api-docs/api/RhinoCommon)

## 源代码通用结构

一般来说，一套**源代码**差不多都是这个样子组织的，可以分散在单独的文件，也可以一个文件梭哈到底

```cs
using System; // 为当前文件声明要使用的命名空间

namespace YourNamespace
{
    class YourClass
    {
    }

    struct YourStruct
    {
    }

    interface IYourInterface
    {
    }

    delegate int YourDelegate();

    enum YourEnum
    {
    }

    namespace YourNestedNamespace
    {
        struct YourStruct
        {
        }
    }
}
```