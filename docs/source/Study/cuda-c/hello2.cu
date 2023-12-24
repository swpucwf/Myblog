#include<stdio.h>
__global__ void hello_from_gpu()
{
   printf("hello word from the gpu!\n");
}

int main()
{
   // 这是表示线程块数量，单个线程块大小
   hello_from_gpu<<<3,4>>>();
   cudaDeviceSynchronize();
   // printf("helloword\n");
   return 0;
}