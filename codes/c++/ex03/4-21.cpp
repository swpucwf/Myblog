#include <iostream>
#include <string>
using namespace std;
//?William Wingate���±������������񡣶���ÿ����������������Ҫ��¼������Ϣ��

//��������˾�����ƣ������ж���������
//        ��������ֱ��
//������������
//        �����һ���ܹ��洢��Щ��Ϣ�Ľṹ������дһ��ʹ�����ֽṹ�����ĳ��򡣳��������û�����������Ϣ��Ȼ����ʾ��Щ��Ϣ����ʹ��cin�������ķ�������cout��
struct Candy {
	string grand;
	float weight;
	int hot_value;
};

int main() {
	auto *caddy = new Candy[3];
	for (int i = 0; i < 3; ++i) {
		caddy[i] = {
				"time",
				5.0,
				2
		};
	}

	for (int i = 0; i < 3; ++i) {
		cout << "=====CandyBar Info=====" << endl;
		cout << "Brand: " << caddy[i].grand << endl;
		cout << "Weight: " << caddy[i].weight << endl;
		cout << "Calorie: " << caddy[i].hot_value << endl;
	}
	return 0;
}
