[TOC]

# for 循环和数组冒泡排序

## 1.for循环

### 1.1 语法格式

```c++
for循环：

	for (表达式1; 表达式2 ; 表达式3)	
	{
		循环体。
	}

	表达式1 --》 表达式2 （判别表达式） --》 为真 --》 循环体。--》 表达式3 --》 表达式2 （判别表达式） --》 为真 --》 循环体 --》 表达式3

	--》 表达式2 （判别表达式）。。。。

```

基本for循环

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// 基础for循环
int main(void)
{
	int i = 0;		// 循环因子
	int sum = 0;

	for (i = 1; i <= 100; i++)
	{
		sum = sum + i;  //sum += i;
	}

	printf("sum = %d\n", sum);

	system("pause");
	return EXIT_SUCCESS;
}
```

![image-20221229225750549](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221229225750549.png)

省略表达式1

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
// 省略表达式1
int main(void)
{
	int i = 1;		// 循环因子
	int sum = 0;

	for (; i <= 100; i++)
	{
		sum = sum + i;  //sum += i;
	}

	printf("sum = %d\n", sum);

	system("pause");
	return EXIT_SUCCESS;
}
```

![image-20221229225957851](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221229225957851.png)

省略表达式3

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// 省略表达式3
int main(void)
{
	int i = 1;		// 循环因子
	int sum = 0;

	for (; i <= 100; )
	{
		sum = sum + i;  //sum += i;
		i++;
	}

	printf("sum = %d\n", sum);

	system("pause");
	return EXIT_SUCCESS;
}

// 省略表达式123
int main0104(void)
{
	int i = 0;		// 循环因子

	//for (;1;)		// 死循环。while(k=1)
	for (;;)
	{
		printf("i = %d\n", i);
		i++;
	}

	system("pause");
	return EXIT_SUCCESS;
}

// 表达式有多个
int main0105(void)
{
	int i = 0;
	int a = 0;

	for (i = 1, a = 3; a < 20; i++)
	{
		printf("i = %d\n", i);
		printf("a = %d\n", a);
		a += 5;
	}
	system("pause");
	return EXIT_SUCCESS;
}


```

![image-20221229225936672](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221229225936672.png)

### 1.2 循环因子

	定义在for之外，for循环结束，也能使用。
	
	定义在for之内，for循环结束，不能使用。
### 1.3 备注

for的3个表达式，均可变换、省略。但，2个分号不能省！

	for (i = 1, a = 3;i < 10, a < 20; i++, a+=5)
	{
		printf("i = %d\n", i);
		printf("a = %d\n", a);
		a += 5;
	}
	
	for(;;) == while(1)  无限循环
### 1.4 猜数字游戏

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>



int main(void)
{
	srand(time(NULL));	//种随机数种子。

	int n = 0;
	int num = rand() % 100;  // 生成随机数

	for (;;)  // while(1)
	{
		printf("请输入猜测的数字：");
		scanf("%d", &n);
		if (n < num)
		{						// for、while、if 如果执行语句只有一条。 { } 可以省略
			printf("猜小了\n");
		}
		else if (n > num)	
			printf("猜大了\n");	
		else
		{
			printf("猜中！！！\n");
			break;			// 跳出
		}
	}
	printf("本尊是：%d\n", num);

	system("pause");

	return 0;
}

```

![image-20221229230144886](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221229230144886.png)

## 2.跳转语句

### 2.1 break

	作用1： 跳出一重循环。 for、while、do while
	
	作用2： 防止case穿透。 switch 	

跳转语句：

break:【重点】

	作用1： 跳出一重循环。 for、while、do while
	
	作用2： 防止case 穿透。 switch 	

continue：【重点】


	作用：结束【本次】循环， continue关键字，之后的循环体，这本次循环中，不执行。


goto：	【了解】

	1. 设定一个标签
	
	2. 使用“goto 标签名” 跳转到标签的位置。（只在函数内部生效）
### 2.2 嵌套循环格式

	外层循环执行一次，内层循环执行一周。
	
	for（i = 0; i < 24; i++）
	{
		for(j = 0; j< 60; j++)
		{
	
			for（k = 0； k< 60; i++）
			{			
			}
		}
	}
### 2.3 电子表打印

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <Windows.h>

int main(void)
{
	int i, j, k;

	// 小时
	for (i = 0; i < 24; i++)
	{
		// 分钟
		for (j = 0; j < 60; j++)
		{
			// 秒
			for (k = 0; k < 60; k++)
			{
				printf("%02d:%02d:%02d\n", i, j, k);
				Sleep(960);
				system("cls");  // 清屏
			}
		}
	}

	system("pause");
	return EXIT_SUCCESS;
}

```

![image-20221229230334418](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221229230334418.png)

### 2.4 9x9乘法表打印

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// 正序99乘法表
int main(void)
{
	for (size_t i = 1; i <= 9; i++)
	{
		for (size_t j = 1; j <= i; j++)
		{
			printf("%dx%d=%d\t", j, i, j * i);
		}
		printf("\n");
	}
	system("pause");
	return EXIT_SUCCESS;
}

// 倒序 99 乘法表
int main0402(void)
{
	int i, j;

	for (i = 9; i >= 1; i--)		// 行
	{
		for (j = 1; j <= i; j++)		// 列
		{
			printf("%dx%d=%d\t", j, i, j * i);
		}
		putchar('\n');
	}

	system("pause");
	return EXIT_SUCCESS;
}

```

![image-20221229231152200](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221229231152200.png)

### 2.5 continue 的使用

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(void)
{
	for (size_t i = 0; i < 5; i++)
	{
		if (i == 3)
		{
			continue;
		}
		printf("i = %d\n", i);
		printf("============1=========\n");
		printf("============2=========\n");
		printf("=============3========\n");
		printf("============4=========\n");
		printf("=============5========\n");

	}

	system("pause");
	return EXIT_SUCCESS;
}

```

![image-20221229231322481](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221229231322481.png)

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


int main(void)
{
	int num = 5;

	//while (num--)  // 当num自减为 0 时循环终止。  等价于 while (num-- != 0)

	while (num-- != 0) // 当num自减为 0 时循环终止。
	{
		printf("num = %d\n", num);
		if (num == 3)
		{
			continue;
		}
		printf("============1=========\n");
		printf("============2=========\n");
		printf("=============3========\n");
		printf("============4=========\n");
		printf("=============5========\n");
	}

	system("pause");
	return EXIT_SUCCESS;
}

```

![image-20221229231435454](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221229231435454.png)

### 2.6 goto案例

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main0601(void)
{
	printf("============1==========\n");
	printf("============2==========\n");
	goto LABLE;

	printf("============3==========\n");
	printf("============4==========\n");
	printf("============5==========\n");
	printf("============6==========\n");
	printf("============7==========\n");

LABLE:
	printf("============8==========\n");
	printf("============9==========\n");
	printf("============10==========\n");

	system("pause");
	return EXIT_SUCCESS;
}





```

![image-20221229231636622](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221229231636622.png)

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
int main(void)
{
	int i = 0;

	for (i = 0; i < 10; i++)
	{
		if (i == 5)
			goto ABX234;

		printf("i = %d\n", i);
	}

	for (int j = 0; j < 20; j++)
	{
		printf("j = %d\n", j);
	}
ABX234:
	printf("end!");
	system("pause");
	return 0;
}



```

![image-20221229231745792](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221229231745792.png)

## 3. 数组

### 3.1 数组定义：

- 相同数据类型的有序连续存储。

- int arr[10] = {1, 2, 23, 4, 5, 6 , 10, 7, 8, 9};

- 各个元素的内存地址 连续。

- 数组名为地址。是数组首元素的地址。

```c++
arr == &arr[0];
printf("数组大小：%u\n", sizeof(arr));
printf("数组元素的大小：%u\n", sizeof(arr[0]));
printf("数组元素个数：%d\n", sizeof(arr)/ sizeof(arr[0]));
```

	- 数据的第一个元素下标： 0
	- 数据的最后一个元素下标： sizeof(arr)/ sizeof(arr[0]) - 1

### 3.2 数组的初始化

数组初始化

	int arr[12] = { 1, 2 ,4, 6, 76, 8, 90 ,4, 3, 6 , 6, 8 }; 【重点】
	
	int arr[10] = { 1, 2 ,4, 6, 76, 8, 9 };  剩余未初始化的元素，默认 0 值。 【重点】
	
	int arr[10] = { 0 }; 初始化一个全为 0 的数组。【重点】
	
	int arr[] = {1, 2, 4, 6, 8}; 	编译器自动求取元素个数  【重点】
	
	int arr[] = {0};  只有一个元素，值为0
	
	int arr[10]; 
	arr[0] = 5;
	arr[1] = 6;
	arr[2] = 7;	其余元素未被初始化，默认值 随机数。
### 3.3 相关案例

- test01

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(void)
{
	int a = 5, b = 29, c = 10;

	int arr[10] = { 1, 2 ,4, 6, 76, 8, 90 ,4, 3, 6 };  //int a = 109;

	printf("&arr[0] = %p\n", &arr[0]);  // 取数组首元素的地址

	printf("arr = %p\n", arr);		// 数组名

	system("pause");
	return EXIT_SUCCESS;
}

```

![image-20221229232041993](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221229232041993.png)

- test02

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>



int main(void)
{
	int a = 5, b = 29, c = 10;

	int arr[12] = { 1, 2 ,4, 6, 76, 8, 90 ,4, 3, 6 , 6, 8 };  //int a = 109;

	printf("数组大小：%u\n", sizeof(arr));

	printf("数组元素的大小：%u\n", sizeof(arr[0]));

	printf("数组元素个数：%d\n", sizeof(arr) / sizeof(arr[0]));

	system("pause");
	return EXIT_SUCCESS;
}
```

![image-20221229232136454](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221229232136454.png)

- test03

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


// 数组初始化
int main(void)
{
	int arr[10];  //int a = 109;
	arr[0] = 5;
	arr[1] = 6;
	arr[2] = 7;

	int n = sizeof(arr) / sizeof(arr[0]);

	for (size_t i = 0; i < n; i++)
	{
		printf("%d\n", arr[i]);
	}

	system("pause");
	return EXIT_SUCCESS;
}


```

![image-20221229232207321](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221229232207321.png)

- test04

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// 数组元素逆序
int main(void)
{
	int arr[] = { 1, 6, 8, 0, 4, 3, 9, 2 };  // {2, 9, 3, 4, 0, 8, 6, 1}
	int len = sizeof(arr) / sizeof(arr[0]); //数组元素个数

	int i = 0;				// i表示数组的首元素下标
	int j = len - 1;		// 表示数组的最后一个元素下标
	int xijinping = 0;		// 临时变量 

	// 交换 数组元素，做逆序
	while (i < j)
	{
		xijinping = arr[i];		// 三杯水法变量交换
		arr[i] = arr[j];
		arr[j] = xijinping;
		i++;
		j--;
	}
	// 打印交互后的 数组
	for (size_t n = 0; n < len; n++)
	{
		printf("%d ", arr[n]);
	}
	printf("\n");

	system("pause");
	return EXIT_SUCCESS;
}

```

![image-20221229232659804](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221229232659804.png)

- 冒泡排序

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(void)
{
	int xjp[] = { 12, 32, 14, 62, 27, 8, 89 };

	int n = sizeof(xjp) / sizeof(xjp[0]);	// 数组元素个数

	int temp = 0;		// 临时变量

	for (size_t i = 0; i < n; i++)
	{
		printf("%d ", xjp[i]);
	}
	printf("\n");

	// 完成乱序数组的冒泡排序。
	for (size_t i = 0; i < n - 1; i++)		// 外层控制行
	{
		for (size_t j = 0; j < n - 1 - i; j++)	// 内层控制列
		{
			if (xjp[j] > xjp[j + 1])		// 满足条件 三杯水交换
			{
				temp = xjp[j];
				xjp[j] = xjp[j + 1];
				xjp[j + 1] = temp;
			}
		}
	}

	// 打印排序后的数组，确定正确性。
	for (size_t i = 0; i < n; i++)
	{
		printf("%d ", xjp[i]);
	}
	printf("\n");

	system("pause");
	return EXIT_SUCCESS;
}

```

![image-20221229233014517](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221229233014517.png)