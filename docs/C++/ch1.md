### myfirst.cpp

```c++
#include <iostream>
using namespace std;
int main() {
    cout<<"come on ,this is the first time!"<<endl;
    return 0;
}

```

![image-20220128003635855](../images/C++%E5%BC%80%E5%A7%8B%E5%AD%A6%E4%B9%A0/image-20220128003635855.png)

### 命名空间使用

```c++
#include <iostream>
using std::cout;
using std::cin;
using std::endl;
# we can use the other seq : using namespace std;
int main() {
    float num;
    cout<<"come on ,this is the first number:"<<endl;
    cin>>num;
    cout<<"come on ,this is the first time!"<<endl;
    return 0;
}

```

### 2.3 输出

```c++
#include <iostream>
using std::cout;
using std::cin;
using std::endl;
int main() {
    int carrots;
    carrots = 25;
    cout<<"I have ";
    cout<<"carrots:";
    cout<<carrots;
    cout<<endl;
    carrots = carrots-1;
    cout<<"Now!I have ";
    cout<<"carrots:";
    cout<<carrots;
    cout<<endl;
}
// 程序2 
#include <iostream>
#include <cmath>
int main() {
    using namespace std;
    double carrots;

    cout<<"How many carrots do you have:"<<endl;
    cin>>carrots;
    cout<<"Here are two more.";
    carrots =carrots+2;
    cout<<"Now you have carrots:"<<carrots<<endl;
    return 0;
}
```

### 2.4 sqrt.cpp

```c++
#include <iostream>
#include <cmath>
int main() {
    using namespace std;
    double area;
    double side;

    cout<<"Enter the floor area,in square feet,of your home:";
    cin>>area;
    side =sqrt(area);

    cout<<"That's the equivalent of square:"<<side<<"feet to the side"<<endl;
    return 0;
}

```

### 2.5 ourfunc.cpp

- 自定义自己的函数

```c++
#include <iostream>
void simon(int);
int main(){
    using namespace std;
    simon(3);
    cout<<"Pick an integer:";
    int count;
    cin>>count;xQxq
    simon(count);
    cout<<"Done!"<<endl;
    return 0;
}
void simon(int n){

    using namespace std;
    cout<<"Simon say touch you says "<<n<<"\ttimes"<<endl;
}
```

- 有返回值函数

  convert.cpp

  ```c++
  #include <iostream>
  int stonetolb(int);
  int main(){
      using namespace std;
      int stone;
      cin>>stone;
      int pounds = stonetolb(stone);
      cout<<stone<<"stone = ";
      cout<<pounds<<" pounds"<<endl;
      return 0;
  
  }
  
  int stonetolb(int sts){
      return 14*sts;
  }
  ```

  

- 总结

  ![image-20220128095955129](../images/C++%E5%BC%80%E5%A7%8B%E5%AD%A6%E4%B9%A0/image-20220128095955129.png)

