#include <iostream>
using namespace std;
int main() {
	int number1, number2;
	int tempMax, tempMin;
	int sum = 0;
	cout << "�������һ������:" << endl;
	cin >> number1;
	cout << "������ڶ�������:" << endl;
	cin >> number2;
	tempMax = number1 > number2 ? number1 : number2;
	tempMin = number1 < number2 ? number1 : number2;
	for (int i = tempMin; i <= tempMax; ++i) {
		sum += i;
	}
	cout << "��Ϊ:" << sum << endl;

	return 0;
}
