#include <iostream>
using namespace std;
void show_menu();
void show_menu() {
	cout << "Please enter one of the following choices:\n"
		"c) carnivore\tp) pianist\n"
		"t) tree\t\tg) game\n";
}
int main() {
	char choice;
	show_menu();
	//# ���get �Ե��ո�
	cin.get(choice).get();
	while (choice != 'c' && choice != 'p' && choice != 't' && choice != 'g') {
		cout << "Please enter a c, p, t or g:";
		cin.get(choice).get();
	}
	switch (choice) {
	case 'c':
		break;
	case 'p':
		break;
	case 't':
		cout << "A maple is a tree";
		break;
	case 'g':
		break;
	}

	return 0;
}

