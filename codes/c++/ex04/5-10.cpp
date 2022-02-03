//
// Created by SWPUCWF on 2022/2/2.
//

#include <cstring>
#include <iostream>

using namespace std;
const int Max_LEN = 20;
const char Exit_WORD[] = "done";

int main() {
	int word_count = 0;
	char words[Max_LEN];

	cout << "Enter words (to stop, type the word done):" << endl;
	while (strcmp(Exit_WORD, words) != 0) {
		word_count++;
		cin >> words;
		cin.get();
	}
	cout << "You entered a total of " << word_count - 1 << " words." << endl;
	return 0;
}
