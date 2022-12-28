### 字符串输出

字符串：

	C语言中，用双引号引着的一串字符，称之为字符串。一定有一个结束标记'\0'
	
	char ch = ‘A’;  一个字节
	
	"abc"  --> 'a''b''c''\0'
	
	‘a’ 不等价 “a”（'a''\0'）

printf函数：

	%s：打印字符串， 挨着从字符串的第一个字符开始打印，打印到'\0'结束。
	
	%d：打印整数
	
	%c：打印字符
	
	%x：打印16进制数
	
	%u：打印无符号
	
	%m.n: 打印实型时用到，一共有 m 位(整数、小数、小数点)，n位小数。
	
	%0m.nf: 其中 f：表示打印实型，一共有 m 位(整数、小数、小数点)，n位小数。 0：表示不足 m 位时，用0凑够m位。
	
	%%： 显示一个%。 转义字符'\' 对 % 转义无效。
	
	%Ns：显示N个字符的字符串。不足N用空格向左填充。
	
	%0Ns：显示N个字符的字符串。不足N用0向左填充。
	
	%-Ns：显示N个字符的字符串。不足N用空格向右填充。
putchar函数：

	输出一个 字符 到屏幕。
	
	直接使用 ASCII 码。
	
	不能输出字符串。
	
	‘abc’既不是一个有效字符，也不是一个有效字符串。
	
	常用putchar('\n');来打印换行。
	
	printf("\n");

scanf函数：

	从键盘接收用户输入。
	
	1. 接收 整数 %d
	
		int a, b, c;  创建变量空间， 等待接收用户输入。
	
		scanf("%d %d %d", &a, &b, &c);
	
	2. 接收 字符 %c
	
		char a, b, c;
	
		scanf("%c %c %c", &a, &b, &c);
	
	3. 接收 字符串 %s
	
		char str[10];	// 定义一个数组，用来接收用户输入的 字符串。
	
		scanf("%s", str);	// 变量名要取地址传递给 scanf， 数组名本身表示地址，不用 & 符。
	
	接收字符串：
	
		1） scanf 具有安全隐患。如果存储空间不足，数据能存储到内存中，但不被保护。【空间不足不要使用】
	
		2） scanf 函数接收字符串时， 碰到 空格 和 换行 会自动终止。不能使用 scanf 的 %s 接收带有空格的字符串。


​		
	将 #define _CRT_SECURE_NO_WARNINGS  添加到程序 第一行。 解决scanf 4996错误


getchar()函数：

	从键盘获取用户输入的 一个字符。
	
	返回该获取的字符的 ASCII 码。


算数运算符：

	先 * / % 后 + -
	
	除法运算后，得到的结果赋值给整型变量时，取整数部分。
	
	除0 ：错误操作。不允许。
	
	对0取余：错误操作。不允许。
	
	不允许对小数取余。余数不能是 小数。 35 % 3.4;
	
	对负数取余，结果为余数的绝对值。10 % -3;  --》 1

++ 和 --：

	前缀自增、自减：
	
		先自增/自减， 在取值。
	
		int a = 10;
	
		++a;		// a = a+1;
	
	后缀自增、自减:
	
		int a  = 10;
	
		a++;		// a = a+1;
	
		先取值， 再自增/自减。

赋值运算：

	int a = 5;
	
	a += 10;  // a = a+10;
	
	a -= 30;  // a = a-30;
	
	a %= 5;	  // a = a % 5;

比较运算符：

	== 判等。
	
	!= 不等于.
	
	< 小于
	
	<= 小于等于	
	
	> 大于
	
	>= 大于等于


	13 < var < 16; ==> var > 13 && var < 16;

逻辑运算符：

	0为假，非0为真。（1）
	
	逻辑非：!
	
		非真为假， 非假为真。
	
	逻辑与： &&（并且）
	
		同真为真，其余为假。
	
	逻辑或：|| （或）
	
		有真为真。同假为假。

运算符优先级：

	[]() > ++ -- (后缀高于前缀) (强转) sizeof > 算数运算（先乘除取余，后加减）> 
	
	比较运算 > 逻辑运算 > 三目运算（条件运算）> 赋值运算 > 逗号运算	


三目运算符： ? :

	表达式1 ？ 表达式2 : 表达式3
	
	表达式1 是一个判别表达式。 如果为真。整个三目运算，取值表达式2。
	
				    如果为假。整个三目运算，取值表达式3。
	
	默认结合性。自右向左。

类型转换：

	隐式类型转换：
	
		由编译器自动完成。
	
		由赋值产生的类型转换。 小--》大 没问题。 大 --》 小 有可能发生数据丢失。
	
		int r = 3;
	
		float s = 3.14 * r * r;
	
		321:	256 128 64 32 16 8 4 2 1
			1   0   1  0  0  0 0 0 1 
	
		char ch  = 0   1  0  0  0 0 0 1 	
	
	强制类型转换：
	
		语法：	（目标类型）带转换变量
	
			（目标类型）带转换表达式
	
		大多数用于函数调用期间，实参给形参传值。


if分支语句：匹配一个范围.属于模糊匹配.

	if (判别表达式1)
	{
		
		判别表达式为真，执行代码。
	}
	else if(判别表达式2)
	{
		判别表达式1为假，并且判别表达式2，执行代码。
	
	}
	else if(判别表达式3)
	{
		判别表达式1为假，判别表达式2为假，判别表达式3， 执行代码。
	}
	。。。
	else
	{
		以上所有判断表达式都为假， 执行代码。
	}


switch 分支：精确匹配.

	switch(判别表达式)
	{
		
		case 1：
			执行语句1；
			break;			// 防止case穿透
	
		case 2:
			执行语句2;
			break;
	
		case 3:
			执行语句3;
			break;
		...
	
		case N:
			执行语句N;
			break;
	
		default:
			其他情况的统一处理;
			break;
	}
	
	case 穿透：
	
		在一个case分支中如果,没有break;那么它会向下继续执行下一个case分支.


​		
while循环:

	while(条件判别表达式)
	{
	
		循环体.
	}


do while 循环:

	无论如何先执行循环体一次。然后在判断是否继续循环。
	
	do {
	
		循环体
	
	} while (条件判别表达式);




1. 字符串输出

```c++
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(void)
{
	char ch = 'a';

	printf("ch = %c\n", ch);

	char str[20] = "hello world";

	printf("str = %s\n", str);

	system("pause");
	return EXIT_SUCCESS;
}
```

![image-20221228223137366](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228223137366.png)

2. 字符串格式化输出

```c++
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(void)
{
	char str[] = "hello world";

	printf("str = |%-15s|\n", str);

	system("pause");
	return EXIT_SUCCESS;
}
```

![image-20221228223207738](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228223207738.png)

3. 字符串格式化输出

```c++
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(void)
{
	putchar(97);  // 'a' == 97
	putchar('b');
	putchar('c');
	putchar('d');
	putchar('abcZ');

	system("pause");
	return EXIT_SUCCESS;
}
```

![image-20221228223329686](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228223329686.png)

### 格式化输入

#### 1. scanf输入

```c++
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// 获取用户输入 整数
int main(void)
{
	int a;
	scanf("%d", &a);		// &：表示取出变量a的地址。描述a的空间
	printf("a = %d\n", a);

	system("pause");
	return EXIT_SUCCESS;
}
```

![image-20221228223534543](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228223534543.png)

2. scanf输入字符

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


int main(void)
{
	char ch1, ch2, ch3;	//	连续定义同类型多个变量。
	scanf("%c%c%c", &ch1, &ch2, &ch3);	// &：表示取出变量ch的地址。描述a的空间
	printf("ch1 = %c\n", ch1);
	printf("ch2 = %c\n", ch2);
	printf("ch3 = %c\n", ch3);

	system("pause");
	return EXIT_SUCCESS;
}

```

![image-20221228223721817](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228223721817.png)

#### 3. 多个整数输入

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
int main(void)
{
	int a1, a2, a3;	//	连续定义同类型多个变量。

	scanf("%d %d %d", &a1, &a2, &a3);	// &：表示取出变量ch的地址。描述a的空间

	printf("a1 = %d\n", a1);
	printf("a2 = %d\n", a2);
	printf("a3 = %d\n", a3);

	system("pause");
	return EXIT_SUCCESS;
}
```

![image-20221228223815738](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228223815738.png)

#### 4.输入字符数组

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
int main(void)
{
	char a[5];			// 大小为5字节的数组
	scanf("%s", a);		// 接收用户键盘输入，写入数组a中
	printf("a = %s\n", a);

	system("pause");
	return EXIT_SUCCESS;
}
```

![image-20221228224100755](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228224100755.png)

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
int main(void)
{
	char a[50];			// 大小为50字节的数组

	scanf("%s", a);		// 接收用户键盘输入，写入数组a中

	printf("a = %s\n", a);

	system("pause");

	return EXIT_SUCCESS;
}

```

![image-20221228224154896](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228224154896.png)

#### getchar输入

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
int main(void)
{
	char ch;

	ch = getchar();		// 接收用户输入，返回接收到的ASCII码

	printf("ch = %c\n", ch);

	putchar(ch);
	putchar('\n');

	system("pause");

	return EXIT_SUCCESS;
}

```

![image-20221228224253474](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228224253474.png)

#### 算数运算符

1. 数学运算

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(void)
{
	int a = 10;
	int b = 20;

	int c = a * b;

	int d = 34 / 10;  // 0.5

	//int m = 98 / 0;

	printf("d = %d\n", d);

	system("pause");
	return EXIT_SUCCESS;
}
```

![image-20221228224429043](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228224429043.png)

2.a++与++a

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
int main(void)
{
	int a = 10;
	int b = 50;
	printf("a = %d\n", a++);  // 先取值给%d, 在自增
	printf("----a = %d", a);
	printf("b = %d\n", ++b);  // 先自增,再取值。 

	printf("----b = %d\n", b);
	system("pause");
	return EXIT_SUCCESS;
}
```

![image-20221228224633645](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228224633645.png)

3. 逻辑运算符

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(void)
{
	int a = 34;
	int b = 0;

	char str[10] = "hello";

	++str[0];

	printf("a = %d\n", !a);
	printf("b = %d\n", !b);

	printf("======%d\n", a && !b);

	printf("------%d\n", !a || b);

	system("pause");
	return EXIT_SUCCESS;
}

```

![image-20221228224718081](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228224718081.png)

4. 三目运算符

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(void)
{
	int a = 40;
	int b = 4;

	int m = a < b ? 69 : a < b ? 3 : 5;  //

	printf("m = %d\n", m);

	printf("%d\n", a > b ? 69 : 100);

	system("pause");
	return EXIT_SUCCESS;
}

```

![image-20221228224947286](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228224947286.png)

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
int main(void)
{
	int a = 10, b = 20, c = 30;

	int x = (a = 1, c = 5, b = 2);		

	printf("x = %d\n", x);
	printf("a = %d\n", a);
	printf("b = %d\n", b);
	printf("c = %d\n", c);

	system("pause");
	return EXIT_SUCCESS;
}


```

![image-20221228225855561](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228225855561.png)

#### 隐式转换

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// 隐式类型转换。
int main(void)
{
	int a = 321;

	char ch = a;

	printf("ch = %d\n", ch);

	system("pause");
	return EXIT_SUCCESS;
}
```

![image-20221228230319730](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228230319730.png)

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// 强制类型转换
int main(int var)
{
	//int *p = (int *)malloc(100);

	float price = 3.6;
	int weight = 4;

	//double sum = (int)price * weight;

	double sum = (int)(price * weight);

	printf("价格：%lf\n", sum);

	system("pause");
	return EXIT_SUCCESS;
}

```

![image-20221228230353620](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228230353620.png)

#### if 语句

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
int main(void)
{
	int a;
	scanf("%d", &a);

	if (a > 0)
	{
		printf("a > 0\n");
	}
	else
	{
		printf("a <= 0\n");
	}

	system("pause");
	return EXIT_SUCCESS;
}
```

![image-20221228230733114](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228230733114.png)

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

//成绩
int main(void)
{
	int score;		// 100--90 优 90 -- 70 良好 70 -- 60 及格  < 60 差劲

	printf("请输入学生成绩：");
	scanf("%d", &score);

	if (score >= 90 && score <= 100)
	{
		printf("优秀\n");
	}
	else if (score < 90 && score >= 70)
	{
		printf("良好\n");
	}
	else if (score < 70 && score >= 60)
	{
		printf("及格\n");
	}
	else
	{
		printf("不及格\n");
	}

	system("pause");
	return EXIT_SUCCESS;
}
```

![image-20221228230809362](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228230809362.png)

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// 三只小猪秤体重
int main(void)
{
	int pig1, pig2, pig3;

	// if (pig1 > pig2 && pig1 > pig3)
	// pig1 > pig2 ? pig1 : pig2;

	printf("请输入三只小猪的体重:");
	scanf("%d %d %d", &pig1, &pig2, &pig3);

	if (pig1 > pig2)		// 满足，说明pig1最重
	{
		if (pig1 > pig3)
		{
			printf("第一只小猪最重，体重为：%d\n", pig1);
		}
		else
		{
			printf("第3只小猪最重，体重为：%d\n", pig3);
		}
	}
	else
	{
		if (pig2 > pig3)
		{
			printf("第2只小猪最重，体重为：%d\n", pig2);
		}
		else
		{
			printf("第3只小猪最重，体重为：%d\n", pig3);
		}
	}

	system("pause");
	return EXIT_SUCCESS;
}


```

**switch语句**

![image-20221228231125026](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228231125026.png)

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(void)
{
	int score;
	scanf("%d", &score);

	switch (score / 10)
	{
	case 10:		//	 100 -- 90 优秀
	case 9:
		printf("优秀\n");
		break;
	case 8:			//   70 -- 90 良好
	case 7:
		printf("良好\n");
		//break;
	case 6:		   // 70 - 60 及格
		printf("及格\n");
		//break;
	default:
		printf("不及格\n");
		break;
	}

	system("pause");
	return EXIT_SUCCESS;
}

```

![image-20221228231652224](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228231652224.png)

**while 语句**

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


int main(void)
{
	int num = 1;

	while (num <= 100)
	{
		if ((num % 7 == 0) || (num % 10 == 7) || (num / 10 == 7))		// 个位、10位、7的倍数
		{
			printf("敲桌子\n");
		}
		else
		{
			printf("%d\n", num);
		}
		num++;  // 递增
	}

	system("pause");
	return EXIT_SUCCESS;
}

```

![image-20221228232553089](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228232553089.png)

**do while**

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// while 的基础用法
int main1001(void)
{
	int a = 1;
	do
	{
		a++;
		printf("a = %d\n", a);
	} while (a < 10);

	system("pause");
	return EXIT_SUCCESS;
}

// 水仙花数：一个三位数。各个位上的数字的立方和等于本数字。 
int main1002(void)
{
	int a, b, c;
	int num = 100;

	do {
		a = num % 10;		// 个位
		b = num / 10 % 10;	// 十位
		c = num / 100;		// 百位

		if (a*a*a + b*b*b + c*c*c == num)
		{
			printf("%d\n", num);
		}
		num++;

	} while (num < 1000);

	system("pause");
	return EXIT_SUCCESS;
}

```

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// while 的基础用法
int main(void)
{
	int a = 1;
	do
	{
		a++;
		printf("a = %d\n", a);
	} while (a < 10);

	system("pause");
	return EXIT_SUCCESS;
}
```



![image-20221228233114908](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228233114908.png)

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// 水仙花数：一个三位数。各个位上的数字的立方和等于本数字。 
int main(void)
{
	int a, b, c;
	int num = 100;

	do {
		a = num % 10;		// 个位
		b = num / 10 % 10;	// 十位
		c = num / 100;		// 百位

		if (a * a * a + b * b * b + c * c * c == num)
		{
			printf("%d\n", num);
		}
		num++;

	} while (num < 1000);

	system("pause");
	return EXIT_SUCCESS;
}



```

![image-20221228233149179](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228233149179.png)