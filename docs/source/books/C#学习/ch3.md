### 接口

#### 接口的概念以及定义

- 定义程序的协议，描述可以属于任何类或者结构的一组相关行为，包含**方法、属性、事件、索引器**
- 当类或者结构继承来自于接口时，继承成员定义但是不能继承实现。实现接口成员需要类中成员必须是公共的、非静态的。
- 接口可以继承来自于其他接口、可以通过继承的基类或者接口多次继承实现。
- 基类可以使用虚拟成员实现接口成员

#### 接口的实现与继承

接口实现

```c#
 interface ImyInterface
    {
        String ID
        {
            get;
            set;
        }
        string Name
        {
            set;
            get;

        }
        void ShowInfo();

    }
```

总体代码如下：

```c#
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HelloWorld
{
    interface ImyInterface
    {
        String ID
        {
            get;
            set;
        }
        string Name
        {
            set;
            get;

        }
        void ShowInfo();

    }
    class Program : ImyInterface
    {
        string id = "";
        string name = "";
        public string ID
        {
            get
            {
                return id;
            }
            set
            {
                id = value;
            }
        }
        public string Name
        {
            get
            {
                return name;
            }
            set
            {
                name = value;
            }

        }
       
        static void Main(string[] args)
        {
            Program program = new Program();
            ImyInterface myInterface = program;
            myInterface.ID = "TM";
            myInterface.Name = "从入门到入土";
            myInterface.ShowInfo();

        }

        public void ShowInfo()
        {
            Console.WriteLine("编号\t姓名");
            Console.WriteLine(ID + "\t" + Name);
        }
    }
}

```

![image-20221227223345476](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221227223345476.png)

代码案例2 多接口继承

```C#
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HelloWorld
{
    interface IPeople
    {
        string Name { get; set; }
        string Sex { get; set; }

    }
    interface ITeacher : IPeople {
        void teach();
    }
    interface IStudent: IPeople {
        void study();
    }

    class Program:IPeople,ITeacher, IStudent
    {
        String name = "";
        string sex = "";
        public string Name
        {
            get { return name; } 
            set { name = value; }
        }
        public string Sex { 
            get { return sex; } 
            set { sex=value;}
        }
        public void teach()
        {
            Console.WriteLine(Name+""+Sex+"教师");
        }
        public void study()
        {
            Console.WriteLine(Name + "" + Sex + "学生");
        }
        static void Main(string[] args)
        {
            Program program= new Program();
            ITeacher teacher = program;
            teacher.Name = "TM";
            teacher.Sex = "男";
            teacher.teach();
            IStudent student = program;
            student.Name = "c#";
            student.Sex = "男";
            student.study();
        }
    }
}
```

![image-20221227225506468](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221227225506468.png)

##### 备注

- 显示接口成员实现中不能含有修饰符、abstract、virtual、override或者static修饰符
- 显示接口成员属于接口成员，而不是类成员，不能直接使用类对象直接访问，只能通过接口来访问。

### 抽象类与抽象方法

#### 抽象类概述以及声明

- 抽象类用来提供多个派生类可共享的公共定义，它与非抽象类的主要区别在于
  - 抽象类不能直接实例化
  - 抽象类可以包含抽象成员、但是非抽象类不可以
  - 抽象类不可被密封

定义实例

```c#
public abstract class myclass{
    public int i;
    public void method(){
    }
}
```

### 抽象类与抽象方法的使用

```c#
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HelloWorld
{
    public abstract class myClass
    {
        private string id = "";
        private string name = "";
        public string ID { get { return id; } set { id = value; } }
        public string Name { get { return name; } set { name = value; } }
    public abstract void ShowInfo();
    }
    public class DriveClass : myClass
    {
        public DriveClass() { }
        public override void ShowInfo() {
            Console.WriteLine(ID+""+Name);
        }
    }
    class Program
    { 
        static void Main(string[] args)
        {
            DriveClass driveClass= new DriveClass();
            myClass myclass= driveClass;
            myclass.ID = "BH0001";
            myclass.Name = "TM";
            myclass.ShowInfo();
           
        }
    }
}

```

![image-20221227231856472](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221227231856472.png)

### 密封类与密封方法

#### 密封类概述以及声明

- 密封类可以用来限制扩展性、密封类不能被其他类继承；派生类不能重写该成员的实现。
- 使用密封类时需要注意：
  - 静态类
  - 类包含安全敏感信息的继承的受保护成员
  - 类继承多个虚成员，并且密封每个成员的开发和测试开销明显大于密封整个类
  - 类是一个要求使用反射进行快速搜索的属性。

```c#
public sealed class myClass
{
    public int i = 0;
    public void method(){
        Console.WriteLine("密封类");
    }
}
```

### 密封方法概述以及声明

```C#
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HelloWorld
{

    public class myClass1{
       public virtual void Method()
        {
            Console.WriteLine("类中的虚方法");
        }
        
    }
    public sealed class myClass2:myClass1
    {
        public sealed override void Method()
        {
            base.Method();// 密封并重写基类中的新方法
            Console.WriteLine("密封类后中重新的类方法");
        }
    }
    class Program
    { 
        static void Main(string[] args)
        {
            myClass2 cl = new myClass2();
            cl.Method();
        }
    }
}

```

### 密封类与密封方法的使用

```C#
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HelloWorld
{

    public class myClass1{
         public virtual void showInfo()
        {

        }
        
    }
    public sealed class myClass2:myClass1
    {
        private string id = "";
        private string name = "";
        public string ID { get { return id; }set { id = value; } }
        public string Name { get { return name; } set { name = value; } }
        public sealed override void showInfo()
        {
            base.showInfo();
            Console.WriteLine("重写类之后的方法");
            Console.WriteLine(ID+""+Name);
        }
    }
    class Program
    { 
        static void Main(string[] args)
        {
            myClass2 myclass2 = new myClass2();
            myclass2.ID = "bh0001";
            myclass2.Name = "tom";
            myclass2.showInfo();
        }
    }
}
```

![image-20221228173324193](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228173324193.png)