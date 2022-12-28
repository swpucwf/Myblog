### 变量

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
            int ls = 927;
            byte shj = 255;
            Console.WriteLine("ls={0}",ls);
            Console.WriteLine("shj={0}",shj);
        }
    }

}

```

![image-20221226230849260](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221226230849260.png)

```c#
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HelloWorld
{
    class C{
        public int Value = 0;
    }
    class Program
    {
        static void Main(string[] args)
        {

            int v1 = 0;
            int v2 = v1;
            C r1 = new C();
            C r2 = r1;//引用
            r2.Value = 112;
            Console.WriteLine("Values:{0}，{1}",v1,v2);
            Console.WriteLine("Refs:{0}，{1}",r2.Value,r2.Value);
           

        }
    }

}

```

![image-20221226231356286](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221226231356286.png)

##### 引用类型

```c#

```

