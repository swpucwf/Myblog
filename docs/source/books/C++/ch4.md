[TOC]

# 二维数组 字符串 函数

## 1.二维数组

### 1.1 二维数组

	int arr[10] = {1,2,3,5,6,7};
	
	{1,2,3,5,6,7};
	{1,2,3,5,6,7};
	{1,2,3,5,6,7};
	{1,2,3,5,6,7};
### 1.2 定义语法

```txt
	int arr[2][3] = 
		{
		{2, 5, 8},
		{7, 9 10}
		};

	int arr[3][5] = {{2, 3, 54, 56, 7 }, {2, 67, 4, 35, 9}, {1, 4, 16, 3, 78}};

	打印：
		for(i = 0; i < 3; i++)		// 行
		{
			for(j = 0; j <5; j++)   // 列
			{
				printf("%d ", arr[i][j]);
			}
			printf("\n");
		}	
```

### 1.3 二维数组大小

```txt
	数组大小: sizeof(arr);

	一行大小: sizeof(arr[0])： 二维数组的一行，就是一个一维数组。

	一个元素大小:sizeof(arr[0][0])		单位：字节

	行数：row = sizeof(arr)/ sizeof(arr[0])

	列数：col = sizeof(arr[0])/ sizeof(arr[0][0])

地址合一：

	printf("%p\n", arr); == printf("%p\n", &arr[0][0]); == printf("%p\n", arr[0]);

	数组的首地址 == 数组的首元素地址 == 数组的首行地址。
```

### 1.4 二维数组的初始化

1. 常规初始化：

	int arr[3][5] = {{2, 3, 54, 56, 7 }, {2, 67, 4, 35, 9}, {1, 4, 16, 3, 78}};

2. 不完全初始化：

	int arr[3][5] = {{2, 3}, {2, 67, 4, }, {1, 4, 16, 78}};  未被初始化的数值为 0 
	
	int arr[3][5] = {0};	初始化一个 初值全为0的二维数组
	
	int arr[3][5] = {2, 3, 2, 67, 4, 1, 4, 16, 78};   【少见】 系统自动分配行列。

3. 不完全指定行列初始化：

	int arr[][] = {1, 3, 4, 6, 7};  二维数组定义必须指定列值。
	
	int arr[][2] = { 1, 3, 4, 6, 7 };  可以不指定行值。
### 1.5 多维数组

多维数组：【了解】

	三维数组：[层][行][列]
	
	数组类型 数组名[层][行][列];
	
	int arr[3][3][4];
	
	{ {12, 3, 4, 5}
	  {12, 3, 4, 5} },
	
	{ {12, 3, 4, 5}
	  {12, 3, 4, 5} },
	
	{ {12, 3, 4, 5}
	  {12, 3, 4, 5} },
	
	for(i = 0; i < 3; i++)  层
	
		for (j = 0; j < 3; j++)  行
	
			for (k = 0; k<4; k++)  列
	
				printf("%d ", arr[i][j][k]);
	
	4维、5维、6维。。。N维。
### 1.6 相关案例

1. 二维数组定义

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(void)
{
	int arr[3][4] = { {2, 7, 8, 5},
						{75, 8, 9, 8},
						{26, 37, 99, 9} };
	for (size_t i = 0; i < 3; i++)		//行
	{
		for (size_t j = 0; j < 4; j++)  //列
		{
			printf("%d ", arr[i][j]);
		}
		printf("\n");
	}

	printf("数组的大小为：%u\n", sizeof(arr));
	printf("数组行的大小：%u\n", sizeof(arr[0]));
	printf("数组一个元素的大小：%u\n", sizeof(arr[0][0]));

	printf("行数=总大小/一行大小：%d\n", sizeof(arr) / sizeof(arr[0]));
	printf("列数=行大小/一个元素大小：%d\n", sizeof(arr[0]) / sizeof(arr[0][0]));

	printf("arr= %p\n", arr);
	printf("&arr[0] = %p\n", &arr[0][0]);
	printf("arr[0] = %p\n", arr[0]);

	system("pause");
	return EXIT_SUCCESS;
}

```

![image-20221229233703611](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221229233703611.png)

2. 二维数组的初始化

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


// 二维数组的初始化
int main(void)
{
	int arr[][2] = { 1, 3, 4, 6, 7 };

	int row = sizeof(arr) / sizeof(arr[0]);
	int col = sizeof(arr[0]) / sizeof(arr[0][0]);

	for (size_t i = 0; i < row; i++)		//行
	{
		for (size_t j = 0; j < col; j++)  //列
		{
			printf("%d ", arr[i][j]);
		}
		printf("\n");
	}

	printf("数组的大小为：%u\n", sizeof(arr));
	printf("数组行的大小：%u\n", sizeof(arr[0]));
	printf("数组一个元素的大小：%u\n", sizeof(arr[0][0]));

	printf("行数=总大小/一行大小：%d\n", sizeof(arr) / sizeof(arr[0]));
	printf("列数=行大小/一个元素大小：%d\n", sizeof(arr[0]) / sizeof(arr[0][0]));

	printf("arr= %p\n", arr);
	printf("&arr[0] = %p\n", &arr[0][0]);
	printf("arr[0] = %p\n", arr[0]);

	system("pause");
	return EXIT_SUCCESS;
}
```

![image-20221229233803771](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221229233803771.png)

3. test03

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(void)
{
	/*
	56 78 92
	45 67 93
	29 83 88
	93 56 89
	72 83 81
	*/
	int scores[5][3] = { 1, 2, 4, 5, 6, 7, 8, 9 };

	int row = sizeof(scores) / sizeof(scores[0]);
	int col = sizeof(scores[0]) / sizeof(scores[0][0]);

	// 获取 5 名学生、3门功课成绩
	for (size_t i = 0; i < row; i++)
	{
		for (size_t j = 0; j < col; j++)
		{
			scanf("%c", &scores[i][j]);
		}
	}
	// 求一个学生的总成绩
	for (size_t i = 0; i < row; i++) // 每个学生
	{
		int sum = 0;
		for (size_t j = 0; j < col; j++)// 每个学生的成绩
		{
			sum += scores[i][j];
		}
		printf("第%d个学生的总成绩为：%d\n", i + 1, sum);
	}
	//求一门功课的总成绩
	for (size_t i = 0; i < col; i++)  // 第几门功课
	{
		int sum = 0;
		for (size_t j = 0; j < row; j++)  // 每门功课的第几个学生
		{
			sum += scores[j][i];
		}
		printf("第%d门功课的总成绩为：%d\n", i + 1, sum);
	}


	system("pause");
	return EXIT_SUCCESS;
}

```

![image-20221229234222461](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221229234222461.png)

4. 多维数组

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(void)
{
	int a[3][4][2] =
	{
		{
			{1, 2},
			{2, 3},
			{4, 5},
			{5, 6}
		},
		{
			{45, 67},
			{78, 90},
			{12, 6},
			{45, 9}
		},
		{
			{ 45, 67 },
			{ 78, 90 },
			{ 12, 6 },
			{ 45, 9 }
		}
	};

	//int arr[2][3][5] = {1, 2, 4, 5, 6, 7, 8 , 9, 0, 0, 7, 9, 8};
	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < 4; j++)
		{
			for (size_t k = 0; k < 2; k++)
			{
				printf("%d ", a[i][j][k]);
			}
			printf("\n");
		}
		printf("\n\n");
	}

	system("pause");
	return EXIT_SUCCESS;
}

```

![image-20221229234341674](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221229234341674.png)

## 2.字符串

### 2.1 字符数组 和 字符串区别

	字符数组： 
		char str[5] = {'h', 'e', 'l', 'l', 'o'};	
	
	字符串：
		char str[6] = {'h', 'e', 'l', 'l', 'o', '\0'};
	
		char str[6] = "hello";
	
		printf("%s");	使用printf打印字符串的时候，必须碰到 \0 结束。
### 2.2 字符串获取 scanf

	注意：	1）用于存储字符串的空间必须足够大，防止溢出。 char str[5];
	
		2) 获取字符串，%s， 遇到空格 和 \n 终止。
	
	借助“正则表达式”, 获取带有空格的字符串：scanf("%[^\n]", str);
### 2.3 字符串操作函数

	gets： 从键盘获取一个字符串， 返回字符串的首地址。 可以获取带有 空格的字符串。 【不安全】
	
		char *gets(char *s);
	
			参数：用来存储字符串的空间地址。
	
			返回值：返回实际获取到的字符串首地址。


	fgets: 从stdin获取一个字符串， 预留 \0 的存储空间。空间足够读 \n, 空间不足舍弃 \n  【安全】
	
		char *fgets(char *s, int size, FILE *stream);
	
			参1：用来存储字符串的空间地址。
	
			参2：描述空间的大小。
	
			参3：读取字符串的位置。	键盘 --》 标准输入：stdin
	
			返回值：返回实际获取到的字符串首地址。


	puts：将一个字符串写出到屏幕. printf("%s", "hello"); / printf("hello\n"); / puts("hello");   输出字符串后会自动添加 \n 换行符。
	
		int puts(const char *s);	
	
			参1：待写出到屏幕的字符串。
	
			返回值： 成功：非负数 0。 失败： -1.		


	fputs：将一个字符串写出到stdout.输出字符串后， 不添加 \n 换行符。
	
		int fputs(const char * str, FILE * stream);	
	
			参1：待写出到屏幕的字符串。		屏幕 --》标准输出： stdout
	
			参数：写出位置 stdout
	
			返回值： 成功：0。 失败： -1.


	strlen: 碰到 \0 结束。
	
		size_t strlen(const char *s);
	
			参1： 待求长度的字符串
	
			返回：有效的字符个数。
### 2.4 字符串追加

	char str1[] = "hello";
	char str2[] = "world";
	
	char str3[100] = {0};
	
	int i = 0;		// 循环 str1
	while (str1[i] != '\0')
	{
		str3[i] = str1[i];  // 循环着将str1中的每一个元素，交给str3
		i++;
	}					// str3=[h e l l o]
	//printf("%d\n", i);  --> 5
	
	int j = 0;		// 循环 str2
	while (str2[j]) // 等价于 while(str2[j] !='\0') 等价于 while (str2[j] != 0)
	{
		str3[i+j] = str2[j];
		j++;
	}					// str3=[h e l l o w o r l d]
	
	// 手动添加 \0 字符串结束标记
	str3[i + j] = '\0';
### 2.5 相关案例

- 字符串定义

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(void)
{
	char str[6] = { 'h', 'e', 'l', 'l', 'o', '\0' };

	char str2[] = "world";  //  == {'w', 'o', 'r', 'l', 'd', '\0'}

	printf("%s\n", str2);

	system("pause");
	return EXIT_SUCCESS;
}
```

![image-20221229234914136](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221229234914136.png)

- 统计字符串字符出现次数

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>



// 统计字符串中每个字符出现的次数
int main(void)
{
	char str[11] = { 0 };		// helloworld -->  26个英文字母 a-z  a:97 d:100

	// scanf("%s", str);
	for (size_t i = 0; i < 10; i++)
	{
		scanf("%c", &str[i]);
	}

	int count[26] = { 0 };  // 代表26个英文字母出现的次数。 

	for (size_t i = 0; i < 11; i++)
	{
		int index = str[i] - 'a';	// 用户输入的字符在 count数组中的下标值。
		count[index]++;
	}

	for (size_t i = 0; i < 26; i++)
	{
		if (count[i] != 0)
		{
			printf("%c字符在字符串中出现 %d 次\n", i + 'a', count[i]);
		}
	}

	system("pause");
	return EXIT_SUCCESS;
}
```

![image-20221229235251503](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221229235251503.png)

- scanf获取字符串

```C++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(void)
{
	char str[100];

	//scanf("%s", str);
	scanf("%[^\n]s", str);

	printf("%s\n", str);

	system("pause");
	return EXIT_SUCCESS;
}

```

![img](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221229235433734.png)

- 字符串操作函数

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cstring>
//gets
int main(void)
{
	char str[10];
	printf("获取的字符串为：%s\n", gets_s(str,10));
	system("pause");
	return EXIT_SUCCESS;
}
```

![image-20221229235915132](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221229235915132.png)

- 字符串拼接函数

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(void)
{
	char str1[] = "hello";  // [ h e l l o \0 ]
	char str2[] = "world";

	char str3[100];

	int i = 0;		// 循环 str1
	while (str1[i] != '\0')   // '\0' != '\0'
	{
		str3[i] = str1[i];  // 循环着将str1中的每一个元素，交给str3
		i++;
	}					// str3=[h e l l o]
	//printf("%d\n", i);  --> 5

	int j = 0;		// 循环 str2
	while (str2[j]) // 等价于 while(str2[j] !='\0') 等价于 while(str2[j] != 0)
	{
		str3[i + j] = str2[j];
		j++;
	}					// str3=[h e l l o w o r l d]

	// 手动添加 \0 字符串结束标记
	str3[i + j] = '\0';

	printf("str3 = %s\n", str3);

	system("pause");
	return EXIT_SUCCESS;
}

```

![image-20221230001830764](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221230001830764.png)

## 3.函数

### 3.1 函数的作用

	1. 提高代码的复用率
	
	2. 提高程序模块化组织性。

### 3.2 函数分类

	系统库函数： 标准C库。 libc
	
		1. 引入头文件 --- 声明函数
		
		2. 根据函数原型调用。
	
	用户自定义:
	
		除了需要提供函数原型之外，还需要提供函数实现。

随机数：

	1. 播种随机数种子： srand(time(NULL));
	
	2. 引入头文件 #include <stdlib.h>  <time.h>
	
	3. 生成随机数： rand() % 100;

### 3.3 函数定义

	包含 函数原型（返回值类型、函数名、形参列表） 和 函数体（大括号一对， 具体代码实现）
	
	形参列表： 形式参数列表。一定包 类型名 形参名。
	
	int add（int a, int b, int c）
	{
		return a+b+c;
	}
	
	int test(char ch, short b, int arr[], int m)

### 3.4 函数调用

	包含 函数名(实参列表);  
	
	int ret = add(10, 4, 28);
		
	实参(实际参数)： 在调用是，传参必须严格按照形参填充。（参数的个数、类型、顺序）  没有类型描述符
	
	int arr[] = {1, 3, 6};

### 3.5 函数声明

	包含 函数原型（返回值类型、函数名、形参列表） + “;”
	
	要求 在函数调用之前，编译必须见过函数定义。否则，需要函数声明。
	
	int add（int a, int b, int c）；


	隐式声明：【不要依赖】
	
		默认编译器做隐式声明函数时，返回都为 int ，根据调用语句不全函数名和形参列表。


	#include <xxx.h> --> 包含函数的声明


exit函数： #include <stdlib.h>

	return关键字：
	
		返回当前函数调用，将返回值返回给调用者。
	
	exit()函数：
	
		退出当前程序。

### 3.6 多文件编程

	将多个含有不同函数功能 .c 文件模块，编译到一起，生成一个 .exe文件。


	<>包裹的头文件为系统库头文件。
	
	""包裹的头文件为用户自定义头文件。
	防止头文件重复包含：头文件守卫。
	
		1） #pragma once		--- windows中
	
		2） #ifndef __HEAD_H__		<--- head.h
	
		    #define __HEAD_H__
	
			.... 头文件内容
	
		    #endif
### 3.7 相关案例

- test01

```c++
- 函数

​```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void bubble_sort(int arr[]);  // 函数声明
void print_arr(int arr[]);
int add(int a, int b);
int main(void)
{
	printf("add = %d\n", add(2, 6));

	int arr[] = { 54, 5, 16, 34 , 6, 9, 34, 1, 7, 93 };

	bubble_sort(arr);

	print_arr(arr);

	system("pause");

	return EXIT_SUCCESS;   // 底层 调用 _exit(); 做退出
}

void print_arr(int arr[])
{
	for (size_t i = 0; i < 10; i++)
	{
		printf("%d ", arr[i]);
	}
}

void bubble_sort(int arr[])
{
	int i, j, temp;

	for (i = 0; i < 10 - 1; i++)
	{
		for (j = 0; j < 10 - 1 - i; j++)
		{
			if (arr[j] < arr[j + 1])
			{
				temp = arr[j];
				arr[j] = arr[j + 1];
				arr[j + 1] = temp;
			}
		}
	}
}

int add(int a, int b)
{
	return a + b;
}


​```
```

![image-20221230002813517](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221230002813517.png)

- exit函数

```C++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int func(int a, char ch);

int main(void)
{
	int ret = func(10, 'a');

	printf("ret = %d\n", ret);

	system("pause");
	//return EXIT_SUCCESS;
	exit(EXIT_SUCCESS);
}

int func(int a, char ch)
{
	printf("a = %d\n", a);

	printf("ch = %c\n", ch);

	//return 10;
	exit(10);
}


```

![image-20221230003018006](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221230003018006.png)