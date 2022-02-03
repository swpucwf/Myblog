//
// Created by SWPUCWF on 2022/2/2.
//

#include <iostream>
#include <string>
using namespace std;
struct Car {
	string Boss;
	int year{};
};
int main() {
	int car_num;
	Car *cars;

	cout << "How many cars do you wish to catalog?";
	cin >> car_num;
	cin.get();
	cars = new Car[car_num];
	for (int i = 0; i < car_num; ++i) {
		cout << "Car #" << (i + 1) << endl;
		cout << "Please enter the make:";
		getline(cin, cars[i].Boss);
		cout << "Please enter the year made:";
		cin >> cars[i].year;
		cin.get();
	}
	for (int i = 0; i < car_num; ++i) {
		cout << "the boss is :" << cars->Boss << "and the year is :" << cars->year << endl;
	}


	return 0;
}