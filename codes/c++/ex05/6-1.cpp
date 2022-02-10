#include <iostream>
#include <cctype>
using namespace std;
// 编写一个程序，读取键盘输入，直到遇到`@`符号为止，并回显输入（数字除外），同时将大写字符转换为小写，将小写字符转换为大写（别忘了`cctype`函数系列）。
int main() {
	char input_word;
	cout << "请输入你的字符：" << endl;
	cin >> input_word;

	while (input_word != '@') {

		//        printf("1");
		if (islower(input_word)) {
			input_word = toupper(input_word);
		}
		else if (isupper(input_word))
		{
			input_word = tolower(input_word);
		}
		cout << input_word << endl;
		cin >> input_word;
	}
	return 0;
}
