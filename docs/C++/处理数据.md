### 3.1 limits.cpp

```c++
#include <iostream>
#include <climits>
int main(){
    using namespace std;
    int n_int = INT_MAX;
    short n_short = SHRT_MAX;
    long n_long = LONG_MAX;

    long long n_llong = LLONG_MAX;

    cout<<"int is "<<sizeof(int)<<"bytes"<<endl;
    cout<<"short is "<<sizeof(n_short)<<"bytes"<<endl;

    cout<<"long is "<<sizeof(n_long)<<"bytes"<<endl;

    cout<<"longlong int is "<<sizeof(n_llong)<<"bytes"<<endl;

    cout<<"longlong int max_value is "<<n_llong<<endl;
    cout<<"long max_value is "<<n_long<<endl;
    cout<<"short max_value is "<<n_short<<endl;
}
```

- 3.2 exceed.cpp

```c++

#include <iostream>
#define ZERO 0
#include <climits>

int main(){
    using namespace std;
    short sam = SHRT_MAX;
    unsigned short sue = sam;
    cout<<"Sam has"<<sam<<"dollars and Sue has" <<sue;
    cout<<"dollars deposited."<<endl<<"Add $1 to each account"<<endl<<"Now";
    sam+=1;
    sue = sue+1;
    cout<<"Sam has"<<sam<<"dollars and Sue has" <<sue;
    cout<<"dollars deposited."<<endl<<"Add $1 to each account"<<endl<<"Now";
    sam = ZERO;
    sue = ZERO;
    cout<<"Sam has"<<sam<<"dollars and Sue has" <<sue;
    cout<<"dollars deposited."<<endl<<"Add $1 to each account"<<endl<<"Now";

    sam-=1;
    sue = sue-1;
    cout<<"Sam has"<<sam<<"dollars and Sue has" <<sue;
    cout<<"dollars deposited."<<endl<<"Add $1 to each account"<<endl<<"Now";

    return 0;
}
```

- 3.3 hexoct.cpp

  十进制，十六进制，八进制

  ```c++
  #include <iostream>
  int main(){
      using namespace std;
      int chest = 42;
      int waist = 0x42;
      int inseam = 042;
  
      cout<<"chest= "<<chest<<endl;
      cout<<"waist= "<<waist<<endl;
      cout<<"inseam= "<<inseam<<endl;
  
  
      return 0;
  }
  ```

  

- 3.4 hexoct2.cpp

```c++
#include <iostream>
int main(){
    using namespace std;
    int chest = 42;
    int waist = 42;
    int inseam = 42;

    cout<<"chest= "<<chest<<endl;
    //数值输出流
    cout<<hex<<endl;
    cout<<"waist= "<<waist<<endl;
    cout<<oct<<endl;
    cout<<"inseam= "<<inseam<<endl;
    return 0;
}
```

- 3.5 chartype.cpp

  ```c++
  #include <iostream>
  int main(){
      using namespace std;
      char ch;
      cout<<"Enter a character:";
      cin>>ch;
      cout<<"Hola!";
      cout<<"Thank you for the "<<ch<<" character"<<endl;
      return 0;
  }
  ```

  

- 3.7 morechar.cpp

  ```c++
  #include <iostream>
  int main() {
      using namespace std;
      char ch = 'M';
      int i = ch;
      cout<<"The ASCII code for "<<ch<<" is "<<i<<endl;
      cout<<"add one to thr character code:"<<endl;
      ch = ch+1;
      i = ch;
      cout<<"The ASCII code for "<<ch<<" is "<<i<<endl;
      cout<<"Displaying char ch using cout.put(ch):";
      cout.put(ch);
      cout.put('!');
      cout<<endl<<"Done!"<<endl;
      return 0;
  }
  ```

- 3.8   	bondini.cpp

  ```c++
  #include <iostream>
  int main() {
      using namespace std;
      cout<<"\aOperation \"HyperHype\" is now activated!"<<endl;
      cout<<"Enter your agent code:_____\b\b\b\b\b\b"<<endl;
      long code;
      cin>>code;
      cout<<"\aYou entered "<<code<<"...\n";
      cout<<"\aCode verified!Proceed with Plan Z3!\n";
      return 0;
  }
  ```

  

- floatnum.cpp

  ```c++
  #include <iostream>
  int main(){
      using namespace std;
      cout.setf(ios_base::fixed,ios_base::floatfield);
      float tub = 10.0/3.0;
      double mint = 10.0/3.0;
  
      const float million = 1.0e6;
  
      cout<<"tub = "<<tub;
      cout<<", a million tubs  = "<<million*tub;
      cout<<",\nand ten million tubs = ";
      cout<<10*million*tub<<endl;
  
      cout<<"mint = " <<mint<<" and a million mints =";
      cout<<million*mint<<endl;
      return 0;
  }
  ```

  

- 3.9 fltadd.cpp

  浮点运算的缺点，精度低

  ```c++
  #include <iostream>
  int main(){
      using namespace std;
      float a  = 2.34E+22f;
      float b = a+1.0f;
      cout<<"a = "<<a <<endl;
      cout<<"b = "<<b <<endl;
      cout<< b-a <<endl;
  
      return 0;
  }
  ```

  

- 3.10  arith.cpp

```c++
#include <iostream>
int main(){
    using namespace std;
    float hats,heads;
    cout.setf(ios_base::fixed,ios_base::floatfield);
    cout<<"Enter a number: ";
    cin>>hats;
    cout<<"Enter a another number: ";
    cin>>heads;

    cout<<"hats = " <<hats<<endl;
    cout<<"heads = "<<heads<<endl;
    cout<<"=-*/"<<endl;
    cout<<heads+hats<<endl;
    cout<<heads-hats<<endl;
    cout<<heads*hats<<endl;
    cout<<heads/hats<<endl;



    return 0;
}

```

- 3.11 divide.cpp

  ```c++
  #include <iostream>
  int main(){
      using namespace std;
      cout.setf(ios_base::fixed,ios_base::floatfield);
      cout<<"Integer division: 9/5 = "<<9/5<<endl;
      cout<<"Float division: 9.0/5.0 = "<<9.0/5.0<<endl;
      cout<<"Mixed division: 9/5 = "<<9.0/5<<endl;
      cout<<"Mixed division: 9/5 = "<<9/5.0<<endl;
      return 0;
  }
  ```

- 3.12 modulus.cpp

求模运算符

```c++
#include <iostream>
int main(){
    using namespace std;
    const int Lbs_per_stn = 14;
    int lbs;
    cout<<"Enter your weight in pounds:";
    cin>>lbs;
    int stone = lbs/Lbs_per_stn;
    int pounds = lbs%Lbs_per_stn;
    cout<<lbs<<"pounds are "<<stone<<" stone, "<<pounds<<"pounds(s)\n";
    return 0 ;
}
```

- 3.13 类型转换 3.13 assign.cpp

```c++
#include <iostream>
int main(){
    using namespace std;
    cout.setf(ios_base::fixed,ios_base::floatfield);
    float tree = 3;
    int guess(3.9832);
    int debt = 7.2E12;
    cout<<"tree = "<<tree<<endl;
    cout<<"guess = "<<guess<<endl;
    cout<<"debt = "<<debt<<endl;

    return 0 ;
}
```

- 3.14 typecast.cpp

```c++
#include <iostream>
int main(){
    using namespace std;
    int auks,bats,coots;
    auks = 19.99+11.99;

    bats = (int)19.99+(int)11.99;

    coots = int(19.99)+int(11.99);

    cout<<"auks = "<<auks<<",bats = "<<bats;
    cout<<",coots = "<<coots<<endl;

    char ch = 'Z';
    cout<<"The code for "<<ch<<" is ";
    cout<<int(ch)<<endl;
    cout<<"Yes, the code is ";
    cout<<static_cast<int>(ch)<<endl;
    return 0 ;
}
```

- 编程练习

  - 3.7.1

  ```c++
  #include <iostream>
  using namespace std;
  float const ratio = 0.2;
  int main(){
      float height;
      cout<<"请输入你的身高_____\b\b\b";
      cin>>height;
  
      cout<<"转换英寸:"<<height*ratio<<endl;
      cout<<"转换英尺:"<<height*height*2*ratio<<endl;
      return 0;
  }
  ```

  - 3.7.2

  ```c++
  #include <iostream>
  using namespace std;
  float cal_bmi(float height,float weights);
  float const ratio_2_meter = 0.0254;
  float const pounds_2_1 = 2.2;
  float const ratio_1_2 = 12;
  
  float cal_bmi(float height,float weights){
      if(height<=0){
          return 0;
      }
      return weights/height/height;
  }
  int main(){
      float pounds,index_1,index_2;
      float temp_height,temp_weights,bmi_ratio;
      cout<<"请输入有多少英尺:";
      cin>>index_1;
      cout<<"请输入有多少英寸:";
      cin>>index_2;
      cout<<"请输入有多少磅:" ;
      cin>>pounds;
      temp_height = ratio_1_2*index_1*ratio_2_meter+index_2*ratio_2_meter;
      temp_weights =pounds/pounds_2_1;
      bmi_ratio = cal_bmi(temp_height,temp_weights);
  
      cout<<"bmi_ratio:" <<bmi_ratio;
  
  }
  ```

  - 3.7.3

  ```c++
  #include <iostream>
  using namespace std;
  const float  ratio_1_2 = 60.0;
  int main(){
      float  minute,index_1,index_2;
      cout<<"请输入有多少度:";
      cin>>index_1;
      cout<<"请输入有多少分:";
      cin>>index_2;
      cout<<"请输入有多少秒:";
      cin>>minute;
  
      cout<<endl;
      float temp1,temp2,temp3,result;
      temp1 = index_1;
      temp2 =index_2/ratio_1_2;
      temp3 = minute/ratio_1_2;
      result = temp2+temp1+temp3;
  
      cout<<result<<endl;
      return 0;
  }
  ```

  - 3.7.4

  ```c++
  #include <iostream>
  
  using namespace std;
  int main(){
      long long seconds,minutes,hours,days;
      cout<<"seconds:";
      cin>>seconds;
  
      cout<<"minutes:";
      minutes = seconds/60;
      cout<<minutes<<endl;
      cout<<"hours:";
      hours = minutes/60;
      cout<<hours<<endl;
      days = hours/60;
      cout<<"days:";
      cout<<days<<endl;
  }
  ```

  

  - 3.7.5

  ```c++
  ```

  