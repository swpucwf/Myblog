#include <iostream>
#include <cstring>
using namespace std;

const int LEN = 20;
const int MAX_LENGTH = 40;
int main() {
    char  FirstName[LEN];
    char  LastName[LEN];
    char  *name = new char [MAX_LENGTH];
    cout << "Enter your first name:";
    cin.getline(FirstName, LEN);
    cout << "Enter your last name:";
    cin.getline(LastName, LEN);

    strcpy(name, LastName);
    strcat(name, ", ");
    strcat(name, FirstName);


    cout << "Here's the information in a single string: " << name << endl;
    return 0;
}
