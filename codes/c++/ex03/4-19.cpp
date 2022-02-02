#include <iostream>
#include <string>
using namespace std;
// William Wingate从事比萨饼分析服务。对于每个披萨饼，他都需要记录下列信息：

//披萨饼公司的名称，可以有多个单词组成
//        披萨饼的直径
//披萨饼的重量
//        请设计一个能够存储这些信息的结构，并编写一个使用这种结构变量的程序。程序将请求用户输入上述信息，然后显示这些信息。请使用cin（或它的方法）和cout。
struct Candy {
	char name[20];
	float length;
	float weight;
};

int main() {
	Candy pizza = {};
	cout << "Please input the Pizza's information:" << endl;
	cout << "Pizza's Company:";
	cin.getline(pizza.name, 40);
	cout << "Pizza's diameter(inches):";
	cin >> pizza.length;
	cout << "Pizza's weight(pounds):";
	cin >> pizza.weight;

	cout << "\n=====The Pizza's Information=====" << endl;
	cout << "Pizza's Company Name: " << pizza.name << endl;
	cout << "Pizza's Diameter: " << pizza.length << endl;
	cout << "Pizza's Weight: " << pizza.weight << endl;
	return 0;
	return 0;
}
#include <iostream>
#include <string>
using namespace std;
// William Wingate从事比萨饼分析服务。对于每个披萨饼，他都需要记录下列信息：

//披萨饼公司的名称，可以有多个单词组成
//        披萨饼的直径
//披萨饼的重量
//        请设计一个能够存储这些信息的结构，并编写一个使用这种结构变量的程序。程序将请求用户输入上述信息，然后显示这些信息。请使用cin（或它的方法）和cout。
struct Candy {
	char name[20];
	float length;
	float weight;
};

int main() {
	auto *pizza = new Candy;
	cout << "Please input the Pizza's information:" << endl;
	cout << "Pizza's Company:";
	cin.getline(pizza->name, 40);
	cout << "Pizza's diameter(inches):";
	cin >> pizza->length;
	cout << "Pizza's weight(pounds):";
	cin >> pizza->weight;

	cout << "\n=====The Pizza's Information=====" << endl;
	cout << "Pizza's Company Name: " << pizza->name << endl;
	cout << "Pizza's Diameter: " << pizza->length << endl;
	cout << "Pizza's Weight: " << pizza->weight << endl;
	return 0;
}
