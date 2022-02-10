// 编写一个程序，它打开一个文本文件，逐个字符地读取该文件，直到到达文件末尾，然后指出该文件中包含多少个字符。
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

int main() {
	string fileName;
	ifstream inFile;
	cout << "Please input your filename?";
	getline(cin, fileName);

	inFile.open(fileName);

	if (!inFile.is_open()) {
		cout << "打开失败！" << endl;
		exit(EXIT_FAILURE);
	}

	char ch;
	int char_count = 0;
	while (inFile.eof()) {
		inFile >> ch;
		char_count++;
	};

	if (char_count == 0) {
		cout << "th number of ch is zero!" << endl;
	}
	else {
		cout << "The file " << fileName << " contains " << char_count << " characters.\n";
	}

	inFile.close();
	return 0;
}
