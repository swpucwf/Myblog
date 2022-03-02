# 习题8.1

  编写通常接受一个参数（字符串的地址），并打印该字符串的函数。然而，如果提供了第二个参数（`int`类型），且该参数不为0，则该函数打印字符串的次数将为该函数被调用的次数（注意：字符串的打印次数不等于第二个参数的值，而等于函数被调用的次数）。是的，这是一个非常可笑的函数，但它让您能够使用本章介绍的一些技术。在一个简单的程序中使用该函数，以演示该函数是如何工作的。

```c++
#include <iostream>

using namespace std;

void print_string(const char *str, int n = 0);

int main() {
    print_string("Hello, Relph!");
    print_string("Hello, Relph!");
    print_string("Hello, Relph!", 5);
    return 0;
}

void print_string(const char *str, int n) {
    static int call_func_count = 0;

    call_func_count++;
    if (n == 0) {
        cout << "arguments = 0, call_func_count = " << call_func_count << endl;
        cout << str << endl;
    } else {
        cout << "arguments != 0, call_func_count = " << call_func_count << endl;
        for (int i = 0; i < call_func_count; i++) {
            cout << str << endl;
        }
    }
}

```

# 习题8.2

  `CandyBar`结构包含3个成员。第一个成员存储`candy bar`的品牌名称；第二个成员存储`candy bar`的重量（可能有小数）；第三个成员存储`candy bar`的热量（整数）。请编写一个程序，它使用一个这样的函数，即将`CandyBar`的引用、`char`指针、`double`和`int`作为参数，并用最后3个值设置相应的结构成员。最后3个参数的默认值分别为`Millennium`、2.85和350。另外，该程序还包含一个以`CandyBar`的引用为参数，并显示结构内容的函数。请尽可能使用`const`。

```c++
#include <iostream>

using namespace std;

struct CandyBar {
    string brand;
    float weight;
    int calorie;
};

void set_candy(CandyBar &candyBar, string s = "Millennium Munch", float w = 2.85, int c = 350);
void show_candy(const CandyBar &candyBar);

int main() {
    CandyBar cb;
    set_candy(cb);
    show_candy(cb);

    set_candy(cb, "Relph Hu", 2.35, 230);
    show_candy(cb);
    return 0;
}

void set_candy(CandyBar &candyBar, string s, float w, int c) {
    candyBar.brand = s;
    candyBar.weight = w;
    candyBar.calorie = c;
}

void show_candy(const CandyBar &candyBar) {
    cout << "=====CandyBar Information=====\n";
    cout << "Brand: " << candyBar.brand << endl;
    cout << "Weight: " << candyBar.weight << endl;
    cout << "Calorie: " << candyBar.calorie << endl;
}

```

### 习题8.3

 编写一个函数，它接受一个指向`string`对象的引用作为参数，并将该`string`对象的内容转换为大写，为此可使用表6.4描述的函数`toupper()`。然后编写一个程序，它通过使用一个循环让您能够使用不同的输入来测试这个函数，该程序的运行情况如下：



```
Enter a string (q to quit):go away
GO AWAY
Next string (q to quit):good grief!
GOOD GRIEF!
Next string (q to quit):q
Bye.
```

```c++
#include <iostream>
#include <string>

using namespace std;

void uppercase(string &s);

int main() {
    string str;

    cout << "Enter a string (q to quit):";
    getline(cin, str);
    while (str != "q") {
        uppercase(str);
        cout << str << endl;
        cout << "Next string (q to quit):";
        getline(cin, str);
    }
    cout << "Bye." << endl;

    return 0;
}

void uppercase(string &s) {
    for (char &i : s) {
        i = toupper(i);
    }
}
```

### 习题8.4

```c++
#include <iostream>
#include <cstring>

using namespace std;

struct stringy {
    char *str;
    int ct;
};

int main() {
    stringy beany;
    char testing[] = "Reality isn't what it used to be.";
    
    set(beany, testing);
    show(beany);
    show(beany, 2);
    testing[0] = 'D';
    testing[1] = 'u';
    show(testing);
    show(testing, 3);
    show("Done!");
    
    return 0;
}
```

  请提供其中描述的函数和原型，从而完成该程序。注意，应有两个`show()`函数，每个都使用默认参数。请尽可能使用`const`参数。`set()`使用`new`分配足够的空间来存储指定的字符串。这里使用的技术与设计和实现类时使用的相似。（可能还必须修改头文件的名称，删除`using`编译指令，这取决于所用的编译器。）

```c++
#include <iostream>
#include <cstring>

using namespace std;

struct stringy {
    char *str;
    int ct;
};

void set(stringy &sty, char *st);
void show(const stringy &sty, int n = 0);
void show(const string &str, int n = 0);

int main() {
    stringy beany;
    char testing[] = "Reality isn't what it used to be.";

    set(beany, testing);
    show(beany);
    show(beany, 2);
    testing[0] = 'D';
    testing[1] = 'u';
    show(testing);
    show(testing, 3);
    show("Done!");

    return 0;
}

void show(const string &str, int n) {
    if (n == 0) {
        n++;
    }
    for (int i = 0; i < n; i++) {
        cout << str << endl;
    }
}

void show(const stringy &sty, int n) {
    if (n == 0) {
        n++;
    }
    for (int i = 0; i < n; i++) {
        cout << sty.str << endl;
    }
}

void set(stringy &sty, char *st) {
    sty.ct = strlen(st);
    sty.str = new char[sty.ct];
    strcpy(sty.str, st);
}
```

### 习题8.5

编写模板函数`max5()`，它将一个包含5个T类型元素的数组作为参数，并返回数组中最大的元素（由于长度固定，因此可以在循环中使用硬编码，而不必通过参数来传递）。在一个程序中使用该函数，将T替换为一个包含5个`int`值的数组和一个包含5个`double`值的数组，以测试该函数。

```c++
#include <iostream>

using namespace std;

const int SIZE = 5;

template<typename T>
T max5(T st[]);

int main() {
    int arr_i[SIZE] = {1, 3, 2, 5, 4};
    double arr_d[SIZE] = {1.1, 2.4, 1.6, 5.8, 2.3};

    cout << "The max element of int array: " << max5(arr_i) << endl;
    cout << "The max element of double array: " << max5(arr_d) << endl;

    return 0;
}

template<typename T>
T max5(T st[]) {
    T max = st[0];
    for (int i = 0; i < SIZE; i++) {
        if (max < st[i])
            max = st[i];
    }
    return max;
}
```

### 习题8.6

 编写模板函数`maxn()`，它将由一个T类型元素组成的数组和一个表示数组元素数目的整数作为参数，并返回数组中最大的元素。在程序对它进行测试，该程序使用一个包含6个`int`元素的数组和一个包含4个`double`元素的数组来调用该函数。程序还包含一个具体化，它将`char`指针数组和数组中的指针数量作为参数，并返回最长的字符串的地址。如果有多个这样的字符串，则返回其中第一个字符串的地址。使用由5个字符串指针组成的数组来测试该具体化。

```c++
#include <iostream>
#include <cstring>

using namespace std;

template<typename T>
T maxn(T st[], int n);

template<>
char *maxn<char *>(char *st[], int n);

int main() {
    int arr_i[6] = {2, 4, 3, 9, 7, 5};
    double arr_d[4] = {3.3, 10.6, 7.3, 4.45};
    string str[] = {"Hello", "Hello world!"};

    cout << "The max element of int array: " << maxn(arr_i, 6) << endl;
    cout << "The max element of double array: " << maxn(arr_d, 4) << endl;
    cout << "The max element of string: " << maxn(str, 2) << endl;

    return 0;
}

template<typename T>
T maxn(T st[], int n) {
    T max = st[0];
    for (int i = 0; i < n; i++) {
        if (max < st[i]) {
            max = st[i];
        }
    }
    return max;
}

template<>
char *maxn<char *>(char *st[], int n) {
    int max = 0;
    for (int i = 0; i < n; i++) {
        if (strlen(st[max]) < strlen(st[i])) {
            max = i;
        }
    }
    return st[max];
}
```

### 习题8.7

修改程序清单8.14，使用两个名为`SumArray()`的模板函数来返回数组元素的总和，而不是显示数组的内容。程序应显示`thing`的总和以及所有`debt`的总和。

```c++
#include <iostream>

using namespace std;

template<typename T>
// template A
void ShowArray(T arr[], int n);

template<typename T>
// template B
void ShowArray(T *arr[], int n);

template<typename T>
T SumArray(T arr[], int n);

template<typename T>
T SumArray(T *arr[], int n);

struct debts {
    char name[50];
    double amount;
};

int main() {
    int things[6] = {13, 31, 103, 301, 310, 130};
    struct debts mr_E[3] =
            {
                    {"Ima Wolfe", 2400.0},
                    {"Ura Foxe",  1300.0},
                    {"Iby Stout", 1800.0}
            };
    double *pd[3];

    // set pointers to the amount members of the structures in mr_E
    for (int i = 0; i < 3; i++)
        pd[i] = &mr_E[i].amount;

    cout << "Listing Mr. E's counts of things:\n";
    // things is an array of int
    ShowArray(things, 6);  // uses template A
    cout << "Listing Mr. E's debts:\n";
    // pd is an array of pointers to double
    ShowArray(pd, 3);      // uses template B (more specialized)

    cout << "The sum of things: " << SumArray(things, 6) << endl;
    cout << "The sum of pd: " << SumArray(pd, 3) << endl;
    return 0;
}

template<typename T>
void ShowArray(T arr[], int n) {
    cout << "template A\n";
    for (int i = 0; i < n; i++)
        cout << arr[i] << ' ';
    cout << endl;
}

template<typename T>
void ShowArray(T *arr[], int n) {
    cout << "template B\n";
    for (int i = 0; i < n; i++)
        cout << *arr[i] << ' ';
    cout << endl;
}

template<typename T>
T SumArray(T arr[], int n) {
    T sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

template<typename T>
T SumArray(T *arr[], int n) {
    T sum = *arr[0] - *arr[0];
    for (int i = 0; i < n; i++) {
        sum += *arr[i];
    }
    return sum;
}
```