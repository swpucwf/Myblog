## 习题7.1

1.  编写一个程序，不断要求用户输入两个数，直到其中的一个为0。对于每两个数，程序将使用一个函数来计算它们的调和平均数，并将结果返回给`main()`，而后者将报告结果。调和平均数指的是倒数平均值的倒数，计算公式如下：

$$
\text { 调和平均数 }=\frac{2.0 * x * y}{(x+y)}
$$

```c++
#include <iostream>

using namespace std;

double calc_harmonic_mean(double x, double y);

int main() {
    double number1;
    double number2;


    cout << "Please input the two number(0 to quit):";
    cin >> number1;
    cin >> number2;


    while (number1 != 0 && number2 != 0) {
        cout << "The harmonic mean of " << number1 << " and " << number2 << " is ";
        cout << calc_harmonic_mean(number1, number2) << endl;


        cout << "Please input next two number(0 to quit):";
        cin >> number1 >> number2;
    }


}

double calc_harmonic_mean(double x, double y){

  return 2.0*x*y/(x+y);
};
```

### 习题7.2 

1.  编写一个程序，要求用户输入最多10个高尔夫成绩，并将其存储在一个数组中。程序允许用户提早结束输入，并在一行上显示所有成绩，然后报告平均成绩。请使用3个数组处理函数来分别进行输入、显示和计算平均成绩。

```c++
#include <iostream>

using namespace std;

const int SIZE = 10;

int input_score(int arr[], int size);
void display(const int arr[], int size);
double calc_average(const int arr[], int size);

int main() {
    int golf_score[SIZE];
    int count;
    count = input_score(golf_score, SIZE);
    display(golf_score, count);
    cout << "The average scores is " << calc_average(golf_score, count) << endl;

    return 0;
}

int input_score(int arr[], int size) {
    int i = 0;
    int count = 0;
    cout << "Please input the golf scores(-1 to quit)" << endl;
    while (i <=  size) {
        cout << "No." << (i + 1) << " golf score:";
        int value;
        cin >> value;
        cin.get();
        if (value < 0) {
            count = i;
            for (; i < size; i++) {
                arr[i] = 0;
            }
            break;
        } else {
            arr[i++] = value;
        }
    }
    return count;
}

void display(const int arr[], int size) {
    cout << "\nHere are " << size << " times golf scores:" << endl;
    for (int i = 0; i < size; i++) {
        cout << arr[i] << "\t";
    }
    cout << endl;
}

double calc_average(const int arr[], int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return 1.0 * sum / size;
}

```

### 7.3

下面是一个结构声明：

```c++
struct box {
   char maker[40];
   float height;
   float width;
   float length;
   float volume;
};Copy to clipboardErrorCopied
```

a. 编写一个函数，按值传递`box`结构，并显示每个成员的值。
b. 编写一个函数，传递`box`结构的地址，并将`volume`成员设置为其他三维长度的乘积。
c. 编写一个使用这两个函数的简单程序。

```c++
#include <iostream>

using namespace std;

struct box {
    char maker[40];
    float height;
    float width;
    float length;
    float volume;
};

void display_box(box b);

void calc_volume(box *b);

int main() {
    box box_maker = {"China", 12, 12, 12, 0};
    display_box(box_maker);
    calc_volume(&box_maker);

    return 0;
}

void calc_volume(box *b) {
    cout << "=====Set Volume=====" << endl;
    b->volume = b->width * b->height * b->length;
    cout << "The volume is " << b->volume << endl;
}

void display_box(const box b) {
    cout << "=====The Information Of The Box=====" << endl;
    cout << "Maker: " << b.maker << endl;
    cout << "Height: " << b.height << endl;
    cout << "Width: " << b.width << endl;
    cout << "Length: " << b.length << endl;
    cout << "Volume: " << b.volume << endl;
}

```



### 7.4 

 许多州的彩票发行机构都使用如程序清单7.4所示的简单彩票玩法的变体。在这些玩法中，玩家从一组被称为域号码（`field number`）的号码中选择几个。例如，可以从域号码1-47中选择5个号码；还可以从第二个区间（如1~27）选择一个号码（称为特选号码）。要赢得头奖，必须正确猜中所有的号码。中头奖的几率是选中所有域号码的几率与选中特选号码几率的乘积。例如，在这个例子中，中头奖的几率是从47个号码中正确选取5个号码的几率与从27个号码中正确选择1个号码的几率的乘积。请修改程序清单7.4，以计算中得这种彩票头奖的几率。

```c++
#include <iostream>

using namespace std;

long double probability(unsigned numbers, unsigned picks);

int main() {
    int field_number, special_number, choices;
    cout << "Enter the field number of choices and\n"
            "the number of picks allowed:\n";
    cin >> field_number >> choices;
    long double field_prob = probability(field_number, choices);
    cout << "The field number, you have one chance in " << field_prob << " of winning.\n";

    cout << "Enter the special number of choices and\n"
            "the number of picks allowed:\n";
    cin >> special_number >> choices;
    long double special_prob = probability(special_number, choices);
    cout << "The special number, you have one chance in " << special_prob << " of winning.\n";

    cout << "The first prize, you have one chance in " << field_prob * special_prob << " of winning.\n";
    return 0;
}

// the following function calculates the probability of picking picks
// numbers correctly from numbers choices
long double probability(unsigned numbers, unsigned picks) {
    long double result = 1.0;  // here come some local variables
    long double n;
    unsigned p;

    for (n = numbers, p = picks; p > 0; n--, p--)
        result = result * n / p;
    return result;
}

```

### 7.5

定义一个递归函数，接受一个整数参数，并返回该参数的阶乘。前面讲过，3的阶乘写作3!3!，等于3 * 2!3∗2!，依此类推；而0!0!被定义为1。通用的计算公式是，如果n大于零，则n!=n * (n-1)!*n*!=*n*∗(*n*−1)!。在程序中对该函数进行测试，程序使用循环让用户输入不同的值，程序将报告这些值的阶乘。

```c++
#include <iostream>

using namespace std;

long long factorial(int n);

int main() {
    int n;

    cout << "Please input a number:";
    while (cin >> n && n > 0) {
        cout << n << "! = " << factorial(n) << endl;
        cout << "Please input next number(-1 to quit):";
    }

    cout << "Bye!" << endl;
    return 0;
}

long long factorial(int n) {
    if (n == 0) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}

```

### 7.6

编写一个程序，它使用下列函数：
  `fill_array()`将一个`double`数组的名称和长度作为参数。它提示用户输入`double`值，并将这些值存储到数组中。当数组被填满或用户输入了非数字时，输入将停止，并返回实际输入了多少个数字。
  `show_array()`将一个`double`数组的名称和长度作为参数，并显示该数组的内容。
  `reverse_array()`将一个`double`数组的名称和长度作为参数，并将存储在数组中的值的顺序反转。
  程序将使用这些函数来填充数组，然后显示数组；反转数组，然后显示数组；反转数组中除第一个和最后一个元素之外的所有元素，然后显示数组。

```c++
#include <iostream>

using namespace std;

const int SIZE = 20;

int fill_array(double arr[], int size);
void show_array(const double arr[], int size);
void reverse_array(double arr[], int size);

int main() {
    double array[SIZE];

    int size = fill_array(array, SIZE);
    show_array(array, size);
    reverse_array(array, size);
    show_array(array, size);
    reverse_array(&array[1], size - 2);
    show_array(array, size);
    return 0;
}

int fill_array(double arr[], int size) {
    int i = 0;
    double value;

    cout << "Enter the number of array(q to quit):" << endl;

    while (i < size) {
        if (cin >> value) {
            arr[i++] = value;
        } else {
            return i;
        }
    }
    return i;
}

void show_array(const double arr[], int size) {
    cout << "The array's data: " << endl;
    for (int i = 0; i < size; i++) {
        cout << arr[i] << "\t";
    }
    cout << endl;
}

void reverse_array(double arr[], int size) {
    double temp;
    for (int i = 0; i < size / 2; i++) {
        temp = arr[i];
        arr[i] = arr[size - i - 1];
        arr[size - i - 1] = temp;
    }
}

```

### 7.7

修改程序清单7.7中的3个数组处理函数，使用两个指针参数来表示区间。`fill_array()`函数不返回实际读取了多少个数字，而是返回一个指针，该指针指向最后被填充的位置；而其他的函数可以将该指针作为第二个参数，以标识数据结尾。

```c++
#include <iostream>

using namespace std;

const int Max = 5;

// function prototypes
double *fill_array(double *start, double *end);
void show_array(double *start, double *end);
void revalue(double r, double *start, double *end);

int main() {
    double properties[Max];

    double *end = fill_array(properties, properties + Max);
    show_array(properties, end);
    if (end - properties > 0) {
        cout << "Enter revaluation factor: ";
        double factor;
        while (!(cin >> factor))    // bad input
        {
            cin.clear();
            while (cin.get() != '\n')
                continue;
            cout << "Bad input; Please enter a number: ";
        }
        revalue(factor, properties, end);
        show_array(properties, end);
    }
    cout << "Done.\n";
    return 0;
}

double *fill_array(double *start, double *end) {
    double temp;
    double *p;
    for (p = start; p != end; p++) {
        int index = p - start + 1;
        cout << "Enter value #" << index << ":";
        cin >> temp;
        if (!cin) {
            cin.clear();
            while (cin.get() != '\n')
                continue;
            cout << "Bad input; input process terminated.\n";
            break;
        } else if (temp < 0) {
            break;
        }
        *p = temp;
    }
    return p;
}

// the following function can use, but not alter,
// the array whose address is ar
void show_array(double *start, double *end) {
    double *p;
    for (p = start; p != end; p++) {
        int index = p - start + 1;
        cout << "Property #" << index << ": $" << *p << endl;
    }
}

// multiplies each element of ar[] by r
void revalue(double r, double *start, double *end) {
    double *p;
    for (p = start; p != end; p++) {
        *p *= r;
    }
}

```

### 7.8 

  在不使用`array`类的情况下完成程序清单7.15所做的工作。编写两个这样的版本：
a. 使用`const char *`数组存储表示季度名称的字符串，并使用`double`数组存储开支。
b. 使用`const char *`数组存储表示季度名称的字符串，并使用一个结构，该结构只有一个成员（一个用于存储开支的`double`数组）。这种设计与使用`array`类的基本设计类似。

```c++
#include <iostream>

using namespace std;

const int Seasons = 4;
const char *Snames[] = {"Spring", "Summer", "Fall", "Winter"};

void fill(double arr[], int size);

void show(double arr[], int size);

int main() {
    double expenses[Seasons];
    fill(expenses, Seasons);
    show(expenses, Seasons);
    return 0;
}

void fill(double arr[], int size) {
    for (int i = 0; i < size; i++) {
        cout << "Enter " << Snames[i] << " expenses:";
        cin >> arr[i];
    }
}

void show(double arr[], int size) {
    double total = 0.0;
    cout << "\nEXPENSES\n";
    for (int i = 0; i < size; i++) {
        cout << Snames[i] << ": $" << arr[i] << '\n';
        total += arr[i];
    }
    cout << "Total: $" << total << '\n';
}

```

### 7.9 

这个练习让您编写处理数组和结构的函数。下面是程序的框架，请提供其中描述的函数，以完成该程序。

```c++
#include <iostream>

using namespace std;
const int SLEN = 30;
struct Student {
    char fullname[SLEN];
    char hobby[SLEN];
    int ooplevel;
};

/*
getinfo() has tow arguments: a pointer to the first element of
an array of student structures and an int representing the
number of element of the array. The function solicits and
stores data about student. It terminates input upon filling
the array or upon encountering a blank line for the student
name. The function returns the actual number of array elements
filled.
*/
int getinfo(Student pa[], int n);

//display1() take a student structures as an argukment
//and displays its contents
void display1(Student st);

//display2() take the address of student structures as an
//argument and displays the structure's contents
void display2(const Student *ps);

//display3() takes the address of the first element of an array
//of student structures and the number of the array elements as
//arguments and displays the cotents of structures
void display3(const Student pa[], int n);

int main() {
    cout << "Enter the class size: ";
    int class_size;
    cin >> class_size;
    while (cin.get() != '\n')
        continue;
    auto *ptr_stu = new Student[class_size];
    int entered = getinfo(ptr_stu, class_size);
    for (int i = 0; i < class_size; i++) {
        display1(ptr_stu[i]);
        display2(&ptr_stu[i]);
    }
    display3(ptr_stu, entered);
    delete[] ptr_stu;
    cout << "Done\n";
    return 0;
}

```

### 7.10

 设计一个名为`calculate()`的函数，它接受两个`double`值和一个指向函数的指针，而被指向的函数接受两个`double`参数，并返回一个`double`值。`calculate()`函数的类型也是`double`，并返回被指向的函数使用`calculate()`的两个`double`参数计算得到的值。例如，假设`add()`函数的定义如下：

```c++
double add(double x, double y) {
    return x + y;
}Copy to clipboardErrorCopied
```

  则下述代码中的函数调用将导致`calculate()`把2.5和10.4传递给`add()`函数，并返回`add()`的返回值（12.9）：

```c++
double q = calculate(2.5, 10.4, add);Copy to clipboardErrorCopied
```

  请编写一个程序，它调用上述两个函数和至少另一个与`add()`类似的函数。该程序使用循环来让用户成对地输入数字。对于每对数字，程序都使用`calculate()`来调用`add()`和至少一个其他的函数。如果读者爱冒险，可以尝试创建一个指针数组，其中的指针指向`add()`样式的函数，并编写一个循环，使用这些指针连续让`calculate()`调用这些函数。提示：下面是声明这种指针数组的方式，其中包含三个指针：

```c++
double (*pf[3]) (double, double);Copy to clipboardErrorCopied
```

  可以采用数组初始化语法，并将函数名作为地址来初始化这样的数组。

```c++
#include <iostream>

using namespace std;

double add(double x, double y);
double mul(double x, double y);
double calculate(double x, double y, double (*pf)(double x1, double x2));

int main() {
    double num1, num2;

    cout << "Please input tow number:";
    while (cin >> num1 >> num2 && num1 != 0 && num2 != 0) {
        double q = calculate(num1, num2, add);
        cout << "call function add: " << num1 << " + " << num2 << " = " << q << endl;
        double m = calculate(num1, num2, mul);
        cout << "call function mul: " << num1 << " * " << num2 << " = " << m << endl;

        cout << "Please input tow number(0 to quit):";
    }

    cout << "Bye!" << endl;
    return 0;
}

double add(double x, double y) {
    return x + y;
}

double mul(double x, double y) {
    return x * y;
}

double calculate(double x, double y, double (*pf)(double x1, double x2)) {
    return pf(x, y);
}

```

