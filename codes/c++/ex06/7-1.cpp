#include <iostream>

using namespace std;

double calc_harmonic_mean(double x, double y);

int main() {
    double number1;
    double number2;


    cout << "Please input the two number(0 to quit):";
    cin >> number1;
    cin >> number2;


    while (number1 != 0 && number2 != 0) {
        cout << "The harmonic mean of " << number1 << " and " << number2 << " is ";
        cout << calc_harmonic_mean(number1, number2) << endl;


        cout << "Please input next two number(0 to quit):";
        cin >> number1 >> number2;
    }


}

double calc_harmonic_mean(double x, double y){

  return 2.0*x*y/(x+y);
};