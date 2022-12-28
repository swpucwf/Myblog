### 1. HelloWorld

```c#
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
namespace HelloWorld
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello world! ");
        }
    }
}
```

2. 注释

```c#
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
namespace HelloWorld
{
    class Program
    {
        static void Main(string[] args)
        {
            /* 第一种注释
             注释
             */

            // 第二种注释
            Console.WriteLine("Hello world! ");
        }
    }
}

```

### 命名空间

类似于仓库，使用命名空间

```c#
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
// 使用命名空间
using N1;

namespace N1
{
    class A
    {
        // 打印函数
        public void print()
        {

            Console.WriteLine("Hello world! in namespace A!");
        }

    }

}

namespace HelloWorld
{
    class Program
    {
        // main 函数为唯一入口
        static void Main(string[] args)
        {
            // 实例对象
            A oa = new A();
            // 调用方法
            oa.print();
        }
    }
}

```

![image-20221226143132860](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221226143132860.png)

