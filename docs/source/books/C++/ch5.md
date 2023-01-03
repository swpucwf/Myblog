[TOC]

# 指针基础

## 1.指针和内存单元

- 指针： 地址。

- 内存单元： 计算机中内存最小的存储单位。――内存单元。大小一个字节。 每一个内存单元都有一个唯一的编号（数）。

- xxxxxxxxxx11   称这个内存单元的编号为 “地址”。

- 指针变量：存地址的变量。

## 2. 指针定义和使用

### 2.1 指针解引用

- *p ： 将p变量的内容取出，当成地址看待，找到该地址对应的内存空间。

- 如果做左值： 存数据到空间中。

- 如果做右值： 取出空间中的内容。

	int a = 10;
	
	int *p = &a;			int* p;--- windows;	int *p ---Linux       int * p ;
	
	int a, *p, *q, b;
	
	*p = 250;			指针的 解引用。 间接引用。
	
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
		int* p = &a;
		//*p = 2000;
		a = 350;
		//printf("a = %d\n", a);
		printf("*p = %d\n", *p);
		printf("sizeof(int *) = %u\n", sizeof(int*));
		printf("sizeof(short *) = %u\n", sizeof(short*));
		printf("sizeof(char *) = %u\n", sizeof(char*));
		printf("sizeof(long *) = %u\n", sizeof(long*));
		printf("sizeof(double *) = %u\n", sizeof(double*));
		printf("sizeof(void *) = %u\n", sizeof(void*));
		system("pause");
		return EXIT_SUCCESS;
	}
	
	```
	
	![image-20221231194127930](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221231194127930.png)

### 2.2 任意“指针”类型大小

- 指针的大小与类型 无关。 只与当前使用的平台架构有关。   
- 32位：4字节。	 64位： 8字节。

### 2.3 野指针

1. 定义 ： 没有一个有效的地址空间的指针。

		int *p;
		*p = 1000;

2. p变量有一个值，但该值不是可访问的内存区域。
   	

	int *p = 10;
	*p = 2000;

【杜绝野指针】

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// 野指针1
int main(void)
{
	// 野指针
	int *p;// 未初始化
	*p = 2000;
	printf("*p = %d\n", *p);
	system("pause");
	return EXIT_SUCCESS;
}

```

![image-20221231194350691](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221231194350691.png)

### 2.4 空指针

```c++
int *p = NULL;     
#define NULL ((void *)0)
```

*p 时 p所对应的存储空间一定是一个**无效的访问区域**。

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// 野指针2
int main(void)
{
	int m;
	//int *p = 1000;   // 0-255 确定留给操作系统
	int* p = 0x0bfcde0000;

	p = &m;

	*p = 2000;

	printf("*p = %d\n", *p);

	system("pause");
	return EXIT_SUCCESS;
}
```



### 2.5 万能指针/泛型指针（void *）

可以接收任意一种变量地址。但是，在使用【必须】借助“强转”具体化数据类型。

		char ch = 'R';
	
		void *p;  // 万能指针、泛型指针
		
		p = &ch;
	
		printf("%c\n", *(char *)p);
```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(void)
{
	int a = 345;

	char ch = 'R';

	void* p;  // 万能指针、泛型指针
	//p = &a;
	p = &ch;

	printf("%c\n", *(char*)p);


	system("pause");
	return EXIT_SUCCESS;
}

```

![image-20221231200416539](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221231200416539.png)

## 3. const关键字

### 3.1  修饰变量

```c++
const int a = 20;

	int *p = &a;

	*p = 650;

	printf("%d\n", a);

```

### 3.2 修饰指针

1. const int *p;

	可以修改 p
	
	不可以修改 *p。

2. int const *p;

	同上。

3. int * const p;

	可以修改 *p
	
	不可以修改 p。

4 const int *const p;

	不可以修改 p。
	不可以修改 *p。

总结：const 向右修饰，被修饰的部分即为只读。

常用：在函数形参内，用来限制指针所对应的内存空间为只读。

## 4. 指针和数组

### 4.1 数组名 

​		【数组名是地址常量】 --- 不可以被赋值。	 ++ / -- / += / -= / %= / /=  (带有副作用的运算符)

	指针是变量。可以用数组名给指针赋值。 ++ -- 

1. 修饰方法1

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


// 操作数组元素的 4 种方法
int main(void)
{
	int a[] = { 1, 2, 4, 5, 6, 7, 8, 9, 0 };

	int n = sizeof(a) / sizeof(a[0]);

	int* p = a;

	printf("sizeof(a) = %u\n", sizeof(a));
	printf("sizeof(p) = %u\n", sizeof(p));

	for (size_t i = 0; i < n; i++)
	{
		//printf("%d "), a[i];
		//printf("%d ", *(a+i));  // a[i] == *(a+i)
		//printf("%d ", p[i]);
		printf("%d ", *(p + i));  // p[i] = *(p+i)
	}
	printf("\n");
	system("pause");
	return EXIT_SUCCESS;
}

```

![image-20221231200747091](C:/Users/CWF/AppData/Roaming/Typora/typora-user-images/image-20221231200747091.png)

2. 指针操作数组

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


// 使用指针++操作数组元素
int main(void)
{
	int arr[] = { 1, 2, 4, 5, 6, 7, 8, 9, 0 };
	int* p = arr;
	int n = sizeof(arr) / sizeof(arr[0]);

	printf("first p = %p\n", p);

	for (size_t i = 0; i < n; i++)
	{
		printf("%d ", *p);
		p++;  // p = p+1;   一次加过一个int大小。 一个元素。
	}
	putchar('\n');

	printf("last p = %p\n", p);

	system("pause");
	return EXIT_SUCCESS;
}
```

![image-20221231200909546](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221231200909546.png)

3. 范例3

```c++
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


// 综合练习
int main(void)
{
	int arr[10];
	int n = sizeof(arr) / sizeof(arr[0]);
	int* p = arr;

	for (size_t i = 0; i < n; i++)
	{
		*(p + i) = 10 + i;  //*(p + i) == arr[i]
	}						// p 指向数组的首地址。

	for (size_t i = 0; i < n; i++)
	{
		printf("%d ", *p);
		p++;
	}						// p指针指向一块无效的内存区域，p为 野指针。
	printf("\n");

	system("pause");
	return EXIT_SUCCESS;
}



```

![image-20221231200949571](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221231200949571.png)

### 4.2 取数组元素

	int arr[] = {1,3, 5, 7, 8};
	
	int *p = arr;  
	
	arr[i] == *(arr+0) == p[0] == *(p+0)

### 4.3 指针和数组区别

1. 指针是变量。数组名为常量。
2. sizeof(指针) ===> 4字节 / 8字节

3. sizeof(数组) ===> 数组的实际字节数。

### 4.4 指针++ 操作数组

	int arr[] = { 1, 2, 4, 5, 6, 7, 8, 9, 0 };
	int *p = arr;		
	
	for (size_t i = 0; i < n; i++)
	{
		printf("%d ", *p);
		p++;  // p = p+1;   一次加过一个int大小。 一个元素。
	}
	
	p的值会随着循环不断变化。打印结束后，p指向一块无效地址空间(野指针)。
## 5. 指针加减运算

### 5.1 数据类型对指针的作用

#### 5.1.1 间接引用

	决定了从指针存储的地址开始，向后读取的字节数。  （与指针本身存储空间无关。）

#### 5.1.2 加减运算

	决定了指针进行 +1/-1 操作向后加过的字节数。

指针 * / % ： error!!!
指针 +- 整数：

1. 普通指针变量+-整数

```txt
char *p; 打印 p 、 p+1  偏过 1 字节。

short*p; 打印 p 、 p+1  偏过 2 字节。

int  *p; 打印 p 、 p+1  偏过 4 字节。		

```

2. 在数组中+- 整数

short arr[] = {1, 3, 5, 8};

int *p = arr;

p+3;			// 向右(后)偏过 3 个元素

p-2;			// 向前(左)偏过 2 个元素

3. &数组名 + 1

加过一个 数组的大小（数组元素个数 x sizeof（数组元素类型））

1. 指针 +- 指针：

  指针 + 指针： error！！！

  指针 - 指针：

  	1） 普通变量来说， 语法允许。无实际意义。【了解】
  	
  	2） 数组来说：偏移过的元素个数。

2. 指针实现 strlen 函数：

  ```c++
  char str[] = "hello";
  
  char *p = str;
  
  while (*p != '\0')
  {
  	p++;
  }
  
  p-str; 即为 数组有效元素的个数。
  ```

3. 指针比较运算：

  1） 普通变量来说， 语法允许。无实际意义。

  2） 数组来说：	地址之间可以进行比较大小。

  		可以得到，元素存储的先后顺序。

  3） int *p;

      p = NULL;		// 这两行等价于： int *p = NULL;
      
      if (p != NULL)
      
      printf(" p is not NULL");
      
      else 
      printf(" p is NULL");

4. 指针数组：

  一个存储地址的数组。数组内部所有元素都是地址。

  1) 

  	int a = 10;
  	int b = 20;
  	int c = 30;
  	int *arr[] = {&a, &b, &c}; // 数组元素为 整型变量 地址
  2) 

  	int a[] = { 10 };
  	int b[] = { 20 };
  	int c[] = { 30 };
  	
  	int *arr[] = { a, b, c }; // 数组元素为 数组 地址。	

  指针数组本质，是一个二级指针。

  二维数组， 也是一个二级指针。

  ```c++
  #define _CRT_SECURE_NO_WARNINGS
  #include <stdio.h>
  #include <string.h>
  #include <stdlib.h>
  #include <math.h>
  #include <time.h>
  
  // 指针数组1
  int main(void)
  {
  	int a = 10;
  	int b = 20;
  	int c = 30;
  
  	int *p1 = &a;
  	int *p2 = &b;
  	int *p3 = &c;
  
  	int *arr[] = {p1, p2, p3};  // 整型指针数组arr， 存的都是整型地址。
  
  	printf("*(arr[0]) = %d\n", *(*(arr + 0)));  //arr[0] ==  *(arr+0)
  
  	printf("*(arr[0]) = %d\n", **arr);
  
  	system("pause");
  	return EXIT_SUCCESS;
  }
  // 指针数组2
  /*
  int main(void)
  {
  	int a[] = { 10 };
  	int b[] = { 20 };
  	int c[] = { 30 };
  
  	int *arr[] = { a, b, c };  // 整型指针数组arr， 存的都是整型地址。
  
  	printf("*(arr[0]) = %d\n", *(*(arr + 0)));  //arr[0] ==  *(arr+0)
  
  	printf("*(arr[0]) = %d\n", **arr);
  
  	system("pause");
  	return EXIT_SUCCESS;
  }
  */
  
  ```

  

8. 多级指针：

	```c++
	int a = 0;
	
	int *p = &a;  				一级指针是 变量的地址。
	
	int **pp = &p;				二级指针是 一级指针的地址。
	
	int ***ppp = &pp;			三级指针是 二级指针的地址。	
	
	int ****pppp = &ppp;			四级指针是 三级指针的地址。	【了解】
	```

备注：多级指针，不能  跳跃定义！

	对应关系：
	
	ppp == &pp;			三级指针
	
	*ppp == pp == &p; 			二级指针
	
	**ppp == *pp == p == &a				一级指针
	
	***ppp == **pp == *p == a				普通整型变量
	
	*p ： 将p变量的内容取出，当成地址看待，找到该地址对应的内存空间。
	
		如果做左值： 存数据到空间中。
	
		如果做右值： 取出空间中的内容。

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
	int *p = &a;
	int **pp = &p;
	// int **pp = &(&a); 不允许！！
	int ***ppp = &pp;

	printf("***ppp = %d\n", ***ppp);
	printf("**pp = %d\n", **pp);
	printf("*p = %d\n", *p);
	printf("a = %d\n", a);

	system("pause");
	return EXIT_SUCCESS;
}

```

## 6. 指针和函数

- 传值和传址
- 指针做函数参数
- 数组做函数参数
- 指针做函数返回值
- 数组做函数返回值

