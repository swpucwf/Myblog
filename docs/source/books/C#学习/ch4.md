### try…catch语句

```C#
try{
    被监控的代码
}
catch(异常类名 异常变量名){
    
}
```

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
            try
            {
                object obj = null;
                int N = (int)obj;
            }
            catch(Exception ex)
            {
                Console.WriteLine("捕获异常:" + "ex");
            }
        }
    }
}

```

![image-20221228173644361](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228173644361.png)

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
            try
            {
                checked // checked 关键字
                {
                    int lnum1;
                    int lnum2;
                    int Num;
                    lnum1 = 6000000;
                    lnum2 = 6000000;
                    Num = lnum1 * lnum2;
                }
            }
            catch (OverflowException)
            {
                Console.WriteLine("引发OverFlow异常");
            }
        }
    }
}

```



![image-20221228173906165](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228173906165.png)

#### throw 关键字

```c#
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HelloWorld
{
    class test
    {
        public int MyInt(String a,string b)
        {
            int int1;
            int int2;
            int num;
            try
            {
                int1 = int.Parse(a);// int 强制转换44
                int2 = int.Parse(b);
                if (int2 == 0)
                {
                    throw new DivideByZeroException();
                }
                num = int1 / int2;
                return num;
            }
            catch(DivideByZeroException de)
            {
                Console.WriteLine("用零除整数引发异常");
                Console.WriteLine(de.Message);
                return 0;
            }
        }
    }
    class Program
    { 
        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("请输入分子:");
                string str1 = Console.ReadLine();
                Console.WriteLine("请输入分母:");
                string str2 = Console.ReadLine();
                test tt = new test();
                Console.WriteLine("分子除以分母 的值:" + tt.MyInt(str1, str2));
            }
            catch(FormatException)
            {
                Console.WriteLine("请输入数值格式数据");
            }
        }
    }
}

```

### try…catch…finally

```c#
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HelloWorld
{
    class test
    {
        public int MyInt(String a,string b)
        {
            int int1;
            int int2;
            int num;
            try
            {
                int1 = int.Parse(a);// int 强制转换44
                int2 = int.Parse(b);
                if (int2 == 0)
                {
                    throw new DivideByZeroException();
                }
                num = int1 / int2;
                return num;
            }
            catch(DivideByZeroException de)
            {
                Console.WriteLine("用零除整数引发异常");
                Console.WriteLine(de.Message);
                return 0;
            }
        }
    }
    class Program
    { 
        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("请输入分子:");
                string str1 = Console.ReadLine();
                Console.WriteLine("请输入分母:");
                string str2 = Console.ReadLine();
                test tt = new test();
                Console.WriteLine("分子除以分母 的值:" + tt.MyInt(str1, str2));
            }
            catch(FormatException)
            {
                Console.WriteLine("请输入数值格式数据");
            }
            finally
            {
                Console.WriteLine("程序执行完毕!");
            }
        }
    }
}

```

![image-20221228183052217](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228183052217.png)