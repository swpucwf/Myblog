#include <iostream>
#include <string>
using namespace std;
//�ṹCandyBar����3����Ա����һ����Ա�洢���ǿ��Ʒ�ƣ��ڶ�����Ա�洢�ǿ��������������С������
//��������Ա�洢���ǿ�Ŀ�·�ﺬ�����������������Щһ��������������ṹ������һ����Ϊsnack��CandyBar������
//�������Ա�ֱ��ʼ��Ϊ��Mocha Munch����2.3��350����ʼ��Ӧ������snackʱ���С���󣬳�����ʾsnack���������ݡ�
struct Candy {
	string grand;
	float weight;
	int hot_value;
};

int main() {
	Candy candy{
		"time",
		5.0,
		2

	};
	cout << "=====CandyBar Info=====" << endl;
	cout << "Brand: " << candy.grand << endl;
	cout << "Weight: " << candy.weight << endl;
	cout << "Calorie: " << candy.hot_value << endl;
	return 0;
}
