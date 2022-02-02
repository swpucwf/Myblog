#include <iostream>
#include <array>

using namespace std;

void avg(array<float, 3> scores, double &score);

void avg(array<float, 3> scores, double &score) {
	double sum = 0;
	for (float score : scores) {
		sum += score;
	}
	score = sum / scores.size();
};


int main() {
	array<float, 3> scores = { 0, 0, 0 };
	double avg_Score;
	cout << "Please input three record of 40 miles.\n";
	cout << "First record(second):";
	cin >> scores[0];
	cout << "Second record(second):";
	cin >> scores[1];
	cout << "Third record(second):";
	cin >> scores[2];
	avg(scores, avg_Score);

	cout << "the ave value is " << avg_Score << endl;



}
