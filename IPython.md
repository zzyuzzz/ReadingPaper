# IPython概览

相信使用过`python`的人一定对notebook不陌生，他是一个交互式的框架，其内核就是IPython。
许多编辑器都已经默认支持IPython,比如Vscode，只要将文件名后缀改为`.ipynb`即可。
Jupyter的内核也是IPython。IPython所有版本都是开源的

Python 最有用的特性之一是它的交互式解释器。它允许非常快速地测试想法，而无需像大多数编程语言中那样创建测试文件的开销。但是，与标准 Python 发行版一起提供的解释器在扩展交互式使用方面受到一定程度的限制。

IPython 的目标是为交互式和探索性计算创建一个全面的环境。为了支持此目标，IPython 有三个主要组件：

+ 增强的交互式`Python shell`。
+ 一种解耦的双进程通信模型，允许多个客户端连接到计算内核。
+ 交互式并行计算的架构现在是`ipyparallel`包的一部分。

## 安装IPython

```bash
pip install ipython
```

但实际上，IPython集成在Jupyter中，而且Jupyter也是非常常用的库，所以我们一般安装Jupyter，
间接安装IPython。

```bash
pip install jupyter
```

## IPython为交互式计算提供的架构

+ 一个强大的交互式shell。
+ Jupyter 的内核。
+ 支持交互式数据可视化和 GUI 工具包的使用。
+ 灵活、可嵌入的解释器，可加载到您自己的项目中。
+ 易于使用的高性能并行计算工具。


## 增强的交互式 Python shell

1. 提供优于 Python 默认的交互式 shell。IPython 具有许多功能，用于 Tab 自动补全、对象自省、系统 shell 访问、跨会话的命令历史记录检索，以及自己的特殊命令系统，用于在交互式工作时添加功能。它试图成为一个非常有效的环境，既用于 Python 代码开发，也用于使用 Python 对象探索问题（在数据分析等情况下）。
2. 作为可嵌入的、随时可用的解释器，用于您自己的程序。交互式 IPython shell 可以通过从另一个程序内部的单个调用来启动，从而提供对当前命名空间的访问。这对于调试目的以及需要混合使用批处理和交互式探索的情况都非常有用。
3. 提供一个灵活的框架，可以用作与其他系统一起工作的基础环境，并将 Python 作为底层桥接语言。具体来说，像Mathematica、IDL和Matlab这样的科学环境启发了它的设计，但类似的想法在许多领域都很有用。
4. 允许对线程图形工具包进行交互式测试。**IPython 支持通过特殊的线程标志对 GTK、Qt、WX、GLUT 和 OS X 应用程序进行交互式、非阻塞控制**。普通的 Python shell 只能对 Tkinter 应用程序执行此操作。