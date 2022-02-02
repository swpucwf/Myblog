#include <iostream>
#include <string>
using namespace std;
//?William Wingate从事比萨饼分析服务。对于每个披萨饼，他都需要记录下列信息：

//披萨饼公司的名称，可以有多个单词组成
//        披萨饼的直径
//披萨饼的重量
//        请设计一个能够存储这些信息的结构，并编写一个使用这种结构变量的程序。程序将请求用户输入上述信息，然后显示这些信息。请使用cin（或它的方法）和cout。
struct Candy {
	string grand;
	float weight;
	int hot_value;
};

int main() {
	auto *caddy = new Candy[3];
	for (int i = 0; i < 3; ++i) {
		caddy[i] = {
				"time",
				5.0,
				2
		};
	}

	for (int i = 0; i < 3; ++i) {
		cout << "=====CandyBar Info=====" << endl;
		cout << "Brand: " << caddy[i].grand << endl;
		cout << "Weight: " << caddy[i].weight << endl;
		cout << "Calorie: " << caddy[i].hot_value << endl;
	}
	return 0;
}
