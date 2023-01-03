

[TOC]

# 指针和字符串

## 1.指针和函数

### 1.1 栈 帧

​	当函数调用时，系统会在 stack 空间上申请一块内存区域，用来供函数调用，主要存放 形参 和 局部变量（定义在函数内部）。

​	当函数调用结束，这块内存区域自动被释放（消失）。

### 1.2 传值和传址

传值：函数调用期间，实参将自己的值，拷贝一份给形参。 

传址：函数调用期间，实参将地址值，拷贝一份给形参。 【重点】

（地址值 -->在swap函数栈帧内部，修改了main函数栈帧内部的局部变量值）

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int swap(int, int);  // 函数声明
int swap2(int *, int *);

int main(void)
{
	int m = 23;
	int n = 57;

	printf("--before-- m = %d, n = %d\n", m, n);
	// 函数调用
	//swap(m, n);  // m/n 实参

	swap2(&m, &n);

	printf("--after-- m = %d, n = %d\n", m, n);

	system("pause");
	return EXIT_SUCCESS;
}

int swap2(int *a, int *b)	// 形参a、b， 需传地址值
{
	int tmp = 0;
	tmp = *a;
	*a = *b;
	*b = tmp;
	return 0;
}

// 函数定义
int swap(int a, int b)	// a/b 形参
{
	int tmp = 0;

	tmp = a;
	a = b;
	b = tmp;

	return 0;
}
```

1. 指针做函数参数

   ```c++
   int swap2(int *a, int *b);
   
   int swap2(char *a, char *b);
   ```

调用时，传有效的地址值。

2. 数组做函数参数

```c++
void BubbleSort(int arr[10]) == void BubbleSort(int arr[])  == void BubbleSort(int *arr) 
```

传递不再是整个数组，而是数组的首地址（一个指针）。

所以，当整型数组做函数参数时，我们通常在函数定义中，封装2个参数。一个表数组首地址，一个表元素个数。

3. 指针做函数返回值

```c++
int *test_func(int a, int b);
```
指针做函数返回值，不能返回【局部变量的地址值】。

4. 数组做函数返回值

C语言，不允许！！！！  只能写成指针形式。

## 2. 指针和字符串：

	1）
		char str1[] = {'h', 'i', '\0'};			变量，可读可写
	
		char str2[] = "hi";				变量，可读可写
	
		char *str3 = "hi";				常量，只读
	
		//str3变量中，存储的是字符串常量“hi”中首个字符‘h’的地址值。
		str3[1] = 'H';	// 错误！！
		char *str4 = {'h', 'i', '\0'};  // 错误！！！
	2）
		当字符串（字符数组）， 做函数参数时， 不需要提供2个参数。 因为每个字符串都有 '\0'。

### 2.1 数组方式

	int mystrcmp(char *str1, char *str2)
	{
		int i = 0;
	while (str1[i] == str2[i])   // *(str1+i) == *(str2+i)
	{
		if (str1[i] == '\0')
		{
			return 0;			// 2字符串一样。
		}
		i++;
	}
	return str1[i] > str2[i] ? 1 : -1;
	}
### 2.2 指针方式

```c++
int mystrcmp2(char *str1, char *str2)
{
	while (*str1 == *str2)   // *(str1+i) == *(str2+i)
	{
		if (*str1 == '\0')
		{
			return 0;			// 2字符串一样。
		}
		str1++;
		str2++;
	}
	return *str1 > *str2 ? 1 : -1;
}
```

//数组版本

```c++
void mystrcpy(char *src, char *dst)
{
	int i = 0;
	while (src[i] != 0)  // src[i] == *(src+i)
	{
		dst[i] = src[i];
		i++;
	}
	dst[i] = '\0';
}
```

//指针版

```c++
void mystrcpy2(char *src, char *dst)
{
	while (*src != '\0')  // src[i] == *(src+i)
	{
		*dst = *src;
		src++;
		dst++;
	}
	*dst = '\0';
}
```

```c++
char *myStrch(char *str, char ch)
{
	while (*str)
	{
		if (*str == ch)
		{
			return str;
		}
		str++;
	}
	return NULL;
}
// hellowrld --- 'o'
char *myStrch2(char *str, char ch)
{
	int i = 0;
	while (str[i])
	{
		if (str[i] == ch)
		{
			return &str[i];  
		}
		i++;
	}
	return NULL;
}
```

```c++
//字符串去空格。
void str_no_space(char *src, char *dst)
{
	int i = 0;   // 遍历字符串src
	int j = 0;	 // 记录dst存储位置
	while (src[i] != 0)
	{
		if (src[i] != ' ')
		{
			dst[j] = src[i];
			j++;
		}
		i++;
	}
	dst[j] = '\0';
}
// 指针版
void str_no_space2(char *src, char *dst)
{
	while (*src != 0)
	{
		if (*src != ' ')
		{
			*dst = *src;
			dst++;
		}
		src++;
	}
	*dst = '\0';
}
```


带参数main函数：

	无参main函数： 	int main(void) == int main()
	
	带参数的main函数： int main(int argc, char *argv[]) == int main(int argc, char **argv)
	
		参1：表示给main函数传递的参数的总个数。
	
		参2：是一个数组！数组的每一个元素都是字符串 char * 
	
	测试1： 
		命令行中的中，使用gcc编译生成 可执行文件，如： test.exe
	
		test.exe abc xyz zhangsan nichousha 
	
		-->
	
		argc --- 5
		test.exe -- argv[0]
		abc -- argv[1]
		xyz -- argv[2]
		zhangsan -- argv[3]
		nichousha -- argv[4]
	
	测试2：
	
		在VS中。项目名称上 --》右键--》属性--》调试--》命令行参数 --》将 test.exe abc xyz zhangsan nichousha 写入。
	
		-->
	
		argc --- 5
		test.exe -- argv[0]
		abc -- argv[1]
		xyz -- argv[2]
		zhangsan -- argv[3]
		nichousha -- argv[4]


str 中 substr 出现次数：

	strstr函数： 在 str中，找substr出现的位置。
	
	char *strstr(char *str, char *substr)   -- #include <string.h>
	
		参1： 原串
	
		参2： 子串
	
		返回值： 子串在原串中的位置。（地址值）；
	
			 如果没有： NULL

实 现：

```c++
int str_times(char *str, char *substr)
{
	int count = 0;
	char *p = strstr(str, substr);  // "llollollo"
while (p != NULL)
{
	count++;
	p += strlen(substr);	// p = p+strlen(substr) --> "llollo"
	p = strstr(p, substr);	// 返回： "llo"
}
return count;
}
```
字符串处理函数：


	字符串拷贝：
	字符串拼接：
	字符串比较：
	字符串格式化输入、输出：
	
		sprintf():
	
		sscanf():	
	
	字符串查找字符、子串：
	
		strchr()
	
		strrchr()
	
		strstr()
	
	字符串分割：
	
		strtok()
	
	atoi/atof/atol：

