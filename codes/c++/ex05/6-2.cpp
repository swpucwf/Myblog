#include <iostream>
#include <array>
#include <cctype>
using namespace std;
// 编写一个程序，最多将10个donation值读入到一个double数组中（如果您愿意，也可以使用模板类array）。
// 程序遇到非数字输入时将结束输入，并报告这些数字的平均值以及数组中有多少个数字大于平均值。
const int LENGTH = 10;
int main() {
	array<double, LENGTH> donation{};
	int count = 0;
	char number;
	double sum = 0.;
	double avg;
	int new_count = 0;
	cout << "请输入number的值:";
	cin >> number;
	while (isdigit(number) && count < LENGTH) {
		cout << number;
		donation[count] = (double)number - 48;
		cout << donation[count] << endl;
		count++;
		cout << "请输入number的值:";
		cin >> number;
	}
	for (int i = 0; i < donation.size(); ++i) {
		sum += donation[i];
	}
	avg = sum / donation.size();

	for (int i = 0; i < donation.size(); ++i) {
		if (donation[i] > avg) {
			new_count++;
		}
	}

	cout << "all:" << new_count << " number are gather than " << avg << endl;
	return 0;
}
