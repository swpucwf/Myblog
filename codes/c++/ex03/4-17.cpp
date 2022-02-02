#include <iostream>
#include <string>
using namespace std;
//结构CandyBar包含3个成员。第一个成员存储了糖块的品牌；第二个成员存储糖块的重量（可以有小数）；
//第三个成员存储了糖块的卡路里含量（整数）。请白那些一个程序，声明这个结构，创建一个名为snack的CandyBar变量，
//并将其成员分别初始化为“Mocha Munch”、2.3和350。初始化应在声明snack时进行。最后，程序显示snack变量的内容。
struct Candy {
	string grand;
	float weight;
	int hot_value;
};

int main() {
	Candy candy{
		"time",
		5.0,
		2

	};
	cout << "=====CandyBar Info=====" << endl;
	cout << "Brand: " << candy.grand << endl;
	cout << "Weight: " << candy.weight << endl;
	cout << "Calorie: " << candy.hot_value << endl;
	return 0;
}
