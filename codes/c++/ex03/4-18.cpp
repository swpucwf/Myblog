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
	Candy candy[3]{
		"time",
		5.0,
		2
	};
	for (int i = 0; i < 3; ++i) {
		cout << "=====CandyBar Info=====" << endl;
		cout << "Brand: " << candy[i].grand << endl;
		cout << "Weight: " << candy[i].weight << endl;
		cout << "Calorie: " << candy[i].hot_value << endl;
	}
	return 0;
}
