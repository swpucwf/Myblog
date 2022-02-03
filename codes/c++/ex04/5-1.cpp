#include <iostream>
using namespace std;
int main() {
	int number1, number2;
	int tempMax, tempMin;
	int sum = 0;
	cout << "请输入第一个整数:" << endl;
	cin >> number1;
	cout << "请输入第二个整数:" << endl;
	cin >> number2;
	tempMax = number1 > number2 ? number1 : number2;
	tempMin = number1 < number2 ? number1 : number2;
	for (int i = tempMin; i <= tempMax; ++i) {
		sum += i;
	}
	cout << "和为:" << sum << endl;

	return 0;
}
