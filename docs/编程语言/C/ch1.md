# 变量常量和数据类型

## 常量

- 不会变化的数据。不能被修改。

1. “hello”、'A'、-10、3.1415926（浮点常量）	

1. #define PI 3.1415	【强调】：没有分号结束标记。 【推荐】 定义宏： 定义语法： #define 宏名 宏值

1. const int a = 10;	定义语法：const 类型名 变量名 = 变量值。

	const关键字： 被该关键字修饰的变量，表示为只读变量。

## 变量

- 会变化的数据。能被修改。

定义语法：类型名 变量名 = 变量值。（一般方法）

	变量三要素：类型名、变量名、变量值。	int r = 3;	float s = PI*r*r;(变量值是一个表达式)
	
	变量的定义：	int a = 40;
	
	变量的声明：	1) int a;	 没有变量值的变量定义 叫做声明。
	
			2）extern int a; 添加了关键字 extern。
	
	1. 变量定义会开辟内存空间。变量声明不会开辟内存空间。
	
	2. 变量要想使用必须有定义。
	
		当编译器编译程序时，在变量使用之前，必须要看到变量定义。如果没有看到变量定义，编译器会自动找寻一个变量声明提升成为定义。
	
		如果该变量的声明前有 extern 关键字，无法提升。


	【建议】：定义变量时。尽量不要重名。

```c
#include<stdio.h>
#define N 1024

int main(int argc, char const *argv[])
{
	int a = 10;
	printf("这是一个常量N:%d\n",N );
	printf("这是一个变量啊：a:%d\n",a);
	return 0;
}
```



## 标识符

	变量和常量的统称。
	
	命名规则：	1. 通常常量使用大写、变量使用小写。大小写严格区分。
	
			2. 只能使用字母、数组、下划线（_）命名标识符。且，数字不能开头。 a-z/A-Z/0-9/_
	
				int a5ir = 10; ok
	
				int _34F = 6; ok
	
				float s2_i85c = 5.4;  ok
	
				int 98ti_54 = 4;  error.
	
			3. 禁止使用关键字和系统函数作为标识符名称。  main/system/printf/sleep....

## sizeof关键字

不是函数。用来求一个变量、类型的大小。 返回一个 无符号整数。 使用 %u 接收返回值。

	方法1： sizeof（类型名）	-- sizeof(int)
	
	方法2： sizeof(变量名)		--- int a = 20； sizeof(a)
	
	【了解】： sizeof 变量名/类型名		举例1： sizeof int
	
						举例2： sizeof a

## 数据类型

有符号整型：

	signed： 有符号 （超级不常用， 通常省略）：		int a = 10; a = -7;	
	
	int类型：	%d		4 字节			
	
		int 名 = 值;
	
	short类型：	%hd		2 字节
	
		short 名 = 值;		short s1 = 3;
	
	long类型：	%ld		4 字节		(windows: 32/64: 4字节； Linux：32位:4字节， 64位:8字节)	
	
		long 名 = 值;		long len = 6;
	
	long long 类型：%lld		8 字节
	
		long long 名= 值;	long long llen = 70;


无符号整型：

	unsigned： 无符号 		只表示数据量，而没有方向（没有正负）	
	
	unsigned int类型：	%u		4 字节
	
		unsigned int 名 = 值;	
		
		unsigned int a = 40;		
	
	unsigned short类型：	%hu		2 字节
	
		unsigned short 名 = 值;		
	
		unsigned short s1 = 3;
	
	unsigned long类型：	%lu		4 字节 (windows: 32/64: 4字节； Linux：32位:4字节， 64位:8字节)
	
		unsigned long 名 = 值;		
	
		unsigned long len = 6;
	
	unsigned long long 类型：%llu		8 字节
	
		unsigned long long 名 = 值;	
	
		unsigned long long llen = 70;


​	
char字符类型：1字节

	存储一个字符。本质是ASCII码。 ‘A’、‘a’、‘%’、‘#’、‘0’
	
	格式匹配符： %c
	
	‘A’：65
	
	‘a’：97
	
	‘0’：48
	
	‘\n’:10
	
	‘\0’: 0


​	
转义字符：

	‘\’	将普通字符转为 特殊意。 将特殊字符转为本身意。
	
	'\n' 和 ‘\0’


实型（浮点数、小数）：

	float：	单精度浮点型。		4字节
	
		float v1 = 4.345;
	
		%f格式匹配符。 默认保留 6 位小数。
	
	double：双精度浮点型。		8字节		【默认】
	
		double v2 = 5.678;
	
	unsigned float v1 = 4.345;	无符号的 float 数据
	
	unsigned double v2 = 5.678;	无符号的 float 数据
	
	printf("n = %08.3f\n", n);
	
		输出的含义为：显示8位数（包含小数点）， 不足8位用0填充。并且保留3位小数。对第4位做四舍五入。


进制和转换：

	十进制转2进制。	--- 除2反向取余法。 【重点】
	
	十进制转8进制。	--- 除8反向取余法。
	
	十进制转16进制。--- 除16反向取余法。
	
		int a = 56;	-- 111000
	
		int b = 173;    -- 10101101
	
	2进制转10进制。
	
		2^10 = 1024
	
		2^9 = 512
	
		2^8 = 256
	
		2^7 = 128
	
		2^6 = 64
	
		2^5 = 32
	
		2^4 = 16
	
		2^3 = 8
	
		2^2 = 4

8进制：

	8进制转10进制。	
	
		定义8进制数语法：	
	
			056： 零开头，每位数0~7之间。	---- 46
	
			0124： 				---- 84
	
	8进制转2进制。
	
		按421码将每个八进制位展开。
	
			056：5--》 101。 6--》 110  。
	
				101110
	
			05326：5 --》 101。 3--》 011。 2--》 010。 6--》 110
	
	2进制转8进制：
	
		1 010 111 010 110：	012726
	
		自右向左，每3位一组，按421码转换。高位不足三位补0

16进制：

	语法： 以0x开头，每位 取 0-9/A-F/a-f
	
			A -- 10
	
			B -- 11
	
			C -- 12
	
			D -- 13
	
			E -- 14
	
			F -- 15
	16 -- 10:
	
		0x1A:  16+10 = 26
	
		0x13F：15+3x16+256 
	
	16 -- 2:
	
		0x1A:	00011010
	
		0x13F：	000100111111
	
	2 -- 16:
	
		0001 0011 1111:		13F
	
		自右向左，每4位一组，按8421码转换。高位不足三位补0

总结：

	int m = 0x15F4;
	
	int n = 345;
	
	int var = 010011; // 不允许。 不能给变量直接复制 二进制数据。
	
	输出格式：
	
		%d %u %o %x %hd %hu %ld %lu %lld %llu %c %f %lf
	
		%d %u %x %c %s 

存储知识：

	1 bit位  就是一个 二进制位
	
	一个字节 1B = 8bit位。 
	
	1KB = 1024B
		
	1MB = 1024KB
	
	1GB = 1024MB
	
	1TB = 1024GB


源码反码补码：【了解】

	源码：
		43 -> 	00101011
		-43 --> 10101011
	
	反码：		
		43 -> 	00101011
		-43 --> 10101011
			11010100
	
	补码：(现今计算机采用的存储形式)
	
		43 -> 	00101011	： 正数不变
		-43 --> 11010101	： 负数，最高位表符号位， 其余取反+1

43-27 ==》 43 + -27


	人为规定： 10000000 --》 -128

-------------------------------------------------

	char 类型：1字节 8个bit位。 数值位有7个。   
	
		有符号: -2^7 --- 2^7-1  == -2^(8-1) -- 2(8-1) -1  
	
			--》 -128 ~ 127
	
		无符号： 0 ~ 2^8 -1 
	
			--》 0~255
	
		不要超出该数据类型的存储范围。


	short类型：2字节  16bit
	
		有符号: -2^15 --- 2^15-1  == -2^(16-1) -- 2(16-1) -1  
	
			--》 -32768 ~ 32767
	
		无符号： 0 ~ 2^8 -1 
	
			--》 0~65535		


	int 类型：4字节			-2^(32-1) -- 2^(32-1)-1
	
		有符号：
	
			--》 -2147483648 ~ 2147483647	
	
		无符号：		0~2^32 -1 
	
			--》 0~4294967295
	
	long类型：4字节
	
		有符号：
	
			--》 -2147483648 ~ 2147483647	
	
		无符号：		0~2^32 -1 
	
			--》 0~4294967295	
	
	longlong 类型：8字节


		有符号：
			--》 -2^(63) ~ 2^(63)-1	
	
		无符号：		
	
			--》 0~2^63-1

```c
#include <stdio.h>
#define PI 3.1415

int main(int argc, char const *argv[])
{
	const int r = 3;
	float area = PI*r*r;
	float L =  2*PI*r;

	printf("圆形的周长为:%f\n",L );
	printf("圆形的面积为:%f\n",area );
	printf("圆的周长还可以写成：%.2f\n", PI * r * r);
	printf("圆的面积还可以写成：%.2f\n", 2 * PI * r);	// 指定保留小数点后保留2位，对第3位进行4舍五入
	return 0;
}
```

```c

#include <stdio.h>		

int main(void)
{
	int var = 54；			// 定义一个变量 var, 定义的同时指定初值为 54
	
	var = 238;			// 使用变量，给变量var赋新值 238

	printf("var = %d\n", var);	//  使用变量，打印变量var的值到屏幕。

	return 0;
}
```

```c
#include <stdio.h>

int main(void)
{
	int a = 10;			// 定义有符号整数 a, 给a赋初值为 10
	short b = 20;		// 定义有符号整数 b, 给a赋初值为 20
	long c = 30L;		// 定义有符号整数 c, 给a赋初值为 30, 可以简写为 long c = 30;
	long long d = 40LL;	// 定义有符号整数 d, 给a赋初值为 40, 可以简写为 long long d = 40;

	printf("sizeof(a)= %u\n", sizeof(a));
	printf("sizeof(b)= %u\n", sizeof(b));
	printf("sizeof(c)= %u\n", sizeof(c));
	printf("sizeof(d)= %u\n", sizeof(d));

	printf("按类型, int 大小为：%u\n", sizeof(int));
	printf("按类型, short 大小为：%u\n", sizeof(short));
	printf("按类型, long 大小为：%u\n", sizeof(long));
	printf("按类型, long long 大小为：%u\n", sizeof(long long));

	system("pause");

	return EXIT_SUCCESS;
}


```

```c
#include <stdio.h>

int main(void)
{
	unsigned int a = 10u;			// 定义无符号整数 a, 给a赋初值为 10, 可以简写为 unsigned int a = 10;
	unsigned short b = 20u;			// 定义无符号整数 b, 给a赋初值为 20, 可以简写为 unsigned short a = 10;
	unsigned long c = 30Lu;			// 定义无符号整数 c, 给a赋初值为 30, 可以简写为 unsigned long c = 30;
	unsigned long long d = 40LLu;	// 定义无符号整数 d, 给a赋初值为 40, 可以简写为 unsigned long long d = 40;

	printf("sizeof(a)= %u\n", sizeof(a));	//按变量名 求变量 a 的大小
	printf("sizeof(b)= %u\n", sizeof(b));	//按变量名 求变量 b 的大小
	printf("sizeof(c)= %u\n", sizeof(c));	//按变量名 求变量 c 的大小
	printf("sizeof(d)= %u\n", sizeof(d));	//按变量名 求变量 d 的大小

	printf("按类型, unsigned int 大小为：%u\n", sizeof(unsigned int));
	printf("按类型, unsigned short 大小为：%u\n", sizeof(unsigned short));
	printf("按类型, unsigned long 大小为：%u\n", sizeof(unsigned long));
	printf("按类型, unsigned long long 大小为：%u\n", sizeof(unsigned long long));


	return 0;
}
```

```c
#include <stdio.h>

int main(void)
{
	char ch = 'a';
	printf("sizeof(ch) = %u\n", sizeof(ch));
	
	printf("%c\n", 97);		//字符'a'
	printf("%c\n", 65);		//字符'A'

	char A = 'A';		// 定义字符变量 A, 初值为 ‘A’
	char a = 'a';		// 定义字符变量 a, 初值为 ‘a’
	
	printf("a = %d\n", a);		//字符'a'的ASCII的值97
	printf("A = %d\n", A);		//字符'A'的ASCII的值65

	printf("A = %c\n", 'a' - 32); //小写a转大写A
	printf("a = %c\n", 'A' + 32); //大写A转小写a
	
	ch = ' ';
	printf("空格ASCII的值：%d\n", ch);    		//空格ASCII的值 32
	printf("\'\\n\'ASCII的值：%d\n", '\n');  	//换行符ASCII的值 10
	printf("字符\'\\0\'：%d\n", '\0');  	//字符'\0'的ASCII的值 0
	printf("字符\'0\'：%d\n", '0');  		//字符'0'的ASCII的值 48
	
	return 0;
}

```



```c
#include <stdio.h>

int main(void)
{
	char ch;		// 有符号 char 型数据，取值范围 -128~127

	//符号位溢出会导致数的正负发生改变
	ch = 0x7f + 2; 	//因为：0x7f == 0111 1111 == 127,所以 等价于 ch = 127 + 2; 
	printf("%d\n", ch);	
	//	   0111 1111
	//+2后 1000 0001，这是负数补码，其原码为 1111 1111，结果为-127

	//最高位的溢出会导致最高位丢失
	unsigned char ch2;
	ch2 = 0xff+1; 	//因为：0xff == 255 == 1111 1111,所以 等价于 ch2 = 255 + 1; 
	printf("%u\n", ch2);
	//		1111 1111
	//+1后 10000 0000， char只有8位最高位的溢出，结果为0000 0000，十进制为0

	ch2 = 0xff + 2; //因为：0xff == 255 == 1111 1111,所以 等价于 ch2 = 255 + 2; 
	printf("%u\n", ch2);
	//		1111 1111
	//+1后 10000 0001， char只有8位最高位的溢出，结果为0000 0001，十进制为1

	return 0;
}

```

