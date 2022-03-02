# helloworld

## 解决提示窗一闪而过：

```tex
1. 通过 system()函数解决：

	在 return 0；之前 添加 system("pause"); 函数调用。

2. 借助VS工具解决：
	
	在项目上 ---》右键 ---》 属性 ---》 配置属性 ---》 连接器 ---》 系统  ---》 子系统 

	---》 在下拉框中选择“控制台 (/SUBSYSTEM:CONSOLE)”
```

## 编写工具

1. 借助VS编辑工具编写。
	
		创建项目 --》 创建 helloworld.c 源文件 --》 写 helloworld程序 --》Ctrl + F5 执行。 

	2. 借助记事本、gcc编译工具编写。

		gcc编译工具的环境变量配置：

			在QT的安装目录中找 gcc.exe 目录位置。例如： C:\Qt\Qt5.5.0\Tools\mingw492_32\bin
			
			我的电脑 --》属性 --》 高级系统设置 --》 环境变量 --》系统环境变量 --》 path --》 将gcc.exe 目录位置写入到 path的值中。

		使用记事本创建 helloworld.c 文件 ——》 在记事本中写 helloworld 程序 

		--> 使用gcc编译工具 ，在记事本写的 helloworld.c 所在目录中，执行 gcc helloworld.c -o myhello.exe  

		--> 在终端（黑窗口）中，运行 ： myhello.exe

## 注释

单行注释：//	
多行注释：/* 注释内容 */ 

	不允许嵌套使用。 多行中可嵌套单行。

system 函数：

	执行系统命令。如： pause、cmd、calc、mspaint、notepad.....
	
	system("cmd");  system("calc");
	
	清屏命令：cls; system("cls");

## gcc编译4步骤：【重点】

	1. 预处理	-E	xxx.i	预处理文件
	
		gcc -E xxx.c -o xxx.i
	
		1) 头文件展开。 --- 不检查语法错误。 可以展开任意文件。
	
		2）宏定义替换。 --- 将宏名替换为宏值。
	
		3）替换注释。	--- 变成空行
	
		4）展开条件编译 --- 根据条件来展开指令。
	
	2. 编译		-S	xxx.s	汇编文件
	
		gcc -S hello.i -o hello.s
	
		1）逐行检查语法错误。【重点】	--- 整个编译4步骤中最耗时的过程。
	
		2）将C程序翻译成 汇编指令，得到.s 汇编文件。
	
	3. 汇编		-c	xxx.o	目标文件
	
		gcc -c hello.s -o hello.o
	
		1）翻译：将汇编指令翻译成对应的 二进制编码。


	4. 链接		无	xxx.exe	可执行文件。
	
		gcc  hello.o -o hello.exe
	
		1）数据段合并
	
		2）数据地址回填
	
		3）库引入

- 源文档

  ```c
  #include<stdio.h>
  int main(int argc, char const *argv[])
  {
  	//this is a test!
  	// 这是一段注释
  	printf("hello world! %s\n","hello world");
  	return 0;
  }
  ```

  

- 预处理 

  ```shell
   gcc -E helloworld.c -o hello.i
  ```

  展开：

  ![image-20220122013247964](../images/ch.assets/image-20220122013247964.png)

- 编译

  ```c
  gcc -S hello.i -o hello.s
  ```

  ![image-20220122013409049](../images/ch.assets/image-20220122013409049.png)

- 汇编

  ```shell
  gcc -c hello.s -o hello.o
  ```

  ![image-20220122013711353](../images/ch.assets/image-20220122013711353.png)

  - 链接

    ```shell
    gcc hello.o -o hello
    ```

    

![image-20220122012110248](../images/ch.assets/image-20220122012110248.png)





## 调试程序：

	添加行号：
	
		工具--》选项 --》文本编辑器--》C/C++ --》行号 选中。
	
	1. 设置断点。F5启动调试
	
	2. 停止的位置，是尚未执行的指令。
	
	3. 逐语句执行一下条 （F11）：进入函数内部，逐条执行跟踪。
	
	3. 逐过程执行一下条 （F10）：不进入函数内部，逐条执行程序。
	
	4. 添加监视：
	
		调试 -->窗口 -->监视：输入监视变量名。自动监视变量值的变化。

