#include <iostream>
#include <string>
using namespace std;
const string Exit_WORD = "done";

int main() {
	int word_count = 0;
	string words;

	cout << "Enter words (to stop, type the word done):" << endl;
	while (Exit_WORD != words) {
		word_count++;
		getline(cin, words);
	}
	cout << "You entered a total of " << word_count - 1 << " words." << endl;
	return 0;
}
