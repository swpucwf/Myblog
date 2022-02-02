#include <iostream>
#include <string>
using namespace std;

int main() {
	string FirstName;
	string LastName;
	string  name;
	cout << "Enter your first name:";
	getline(cin, FirstName);
	cout << "Enter your last name:";
	getline(cin, LastName);
	name = FirstName + ", " + LastName;
	cout << "Here's the information in a single string: " << name << endl;
	return 0;
}
