### 1. c++ HelloWorld

```c++
#include <iostream>
using namespace std;
int main()
{
    cout << "Hello World!\n";
}

```

![image-20221228214757436](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228214757436.png)

```c++
#include <iostream>
#include <stdio.h>
using namespace std;
#define N 1024
int main()
{
	int a = 10;
	cout << "这是一个常量N：" << N << endl;
	cout << "这是一个常量a：" << a << endl;
	system("pause");
	return 0;
}


```

![image-20221228215216604](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228215216604.png)

```c++
#include <iostream>
#include <stdio.h>
using namespace std;
#define PI 3.1415
int main()
{
	const int r = 3;
	// 圆的面积 = PI x 半径的平方
	float s = PI * r * r;
	// 圆的周长 2*PI*r
	float ll = 2 * PI * r;

	cout << "圆的周长为：" << ll << endl;
	cout << "圆的面积为：" << s << endl;
}


```

![image-20221228215542732](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228215542732.png)

```c++
#include <iostream>
#include <stdio.h>
using namespace std;
#define PI 3.1415
int main()
{
	int var = 54;
	var = 238;
	cout << "var= " << var << endl;
	return 0;
}


```



![image-20221228215634379](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228215634379.png)

### 不同数据类型

```c++
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

![image-20221228220630510](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228220630510.png)

```c++
#include <stdio.h>
#include <iostream>
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

	system("pause");

	return EXIT_SUCCESS;
}
```

![image-20221228221034327](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228221034327.png)

```c++
#include <stdio.h>
#include <iostream>
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
	ch2 = 0xff + 1; 	//因为：0xff == 255 == 1111 1111,所以 等价于 ch2 = 255 + 1; 
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

![image-20221228222444024](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221228222444024.png)