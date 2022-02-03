//
// Created by SWPUCWF on 2022/2/2.
//

#include <iostream>
#include <array>
using namespace std;
#define MAX_SIZE 101
int main() {

	array<long double, MAX_SIZE> fun = {};
	fun[0] = fun[1] = 1;

	for (int i = 1; i <= fun.size(); ++i) {
		fun[i] = i * fun[i - 1];
		cout << i - 1 << "!" << "=" << i * fun[i - 1] << endl;
	}


	for (int i = 0; i < fun.size(); ++i) {
		cout << fun[i] << endl;
	}


	return 0;
}