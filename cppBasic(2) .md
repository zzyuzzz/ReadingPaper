# C++ const、constexpr与consteval作用与区别


在[C++ 常量表达式和编译时优化](cppBasic(1).md)中，我们已经提到了常量、编译时常量与运行时常量的概念。为了加深理解，我们再重新明晰一下这三者的概念。

+ 常量：初始化之后便不可修改的量。在c++中使用const修饰的“变量”称为常量。const修饰的常量可以是编译时常量，也可以是运行时常量。现代编译器将自行推理决断。
    ```c++
    #include<iostream>
    int main(){
        const int x{1};
        std::cout << x << std::endl;//可以读取
        x = 5； //error，编译错误，常量不可修改
    }
    ```
+ 编译时常量：在编译时，编译器将常量的值计算出来，不必等到每次运行时计算。
    ```c++
    #include<iostream>
    int main(){
        const int x{1 + 8};
        std::cout << x << std::endl;
    }
    ```
    就像上面的代码，若不优化，则每次运行编译后的程序时都需要计算$1+8$,如果程序执行一百万次，则$1+8$将被计算一百万次。这完全是没有必要的。
    优化后的结果类似以下代码：
    ```c++
    #include<iostream>
    int main(){
        const int x{9};
        std::cout << x << std::endl;
    }
    ```

+ 运行时常量：程序运行时才确定下来的常量。
    ```c++
    #include<iostream>
    int main(){
        int a = 9;
        const int x{a}; //由于a是一个变量，所以x只能在运行到此处的时候才能确定值。
        std::cout << x << std::endl;
    }
    ```

### constexpr
尽管现代编译器已经可以自主判断是否应该编译时优化，但是由于某些表达式十分复杂，仅仅依赖于编译器是十分困难的。于是c++提供了**constexpr**关键字，它告诉编译器其修饰的常量是编译时常量可以在编译时优化。

**注意：** `constexpr`表示该对象可以在常量表达式中使用。初始值设定项的值在编译时必须已知。`constexpr`对象可以在运行时或编译时进行计算。`constexpr`与**std::string、std::vector和其他使用动态内存分配的类型**不完全兼容。

```c++
#include <iostream>

double nonconstexpr_ex(double radius)
{
    constexpr double pi { 3.14159265359 };
    return 2.0 * pi * radius;
}
constexpr double constexpr_ex(double radius)
{
    constexpr double pi { 3.14159265359 };
    return 2.0 * pi * radius;
}
int main()
{
    constexpr double circumference { nonconstexpr_ex(3.0) }; // 编译错误
    constexpr double circumference { constexpr_ex(3.0) }; // 正确
    std::cout << "Our circle has circumference " << circumference << "\n";
    return 0;
}
```

**注**：
1. `constexpr` 函数也可以在运行时进行计算。所以在非必要时，编译器可自行选择函数是否在编译时计算。
2. `constexpr` 函数是隐式内联的, 编译器必须能够看到`constexpr`（或 `consteval`）函数的完整定义，而不仅仅是前向声明。

### consteval

`consteval`是**c++20**引入的关键字。用于指示函数必须在编译时计算，否则将导致编译错误。此类函数称为即时函数（immediate functions）。
