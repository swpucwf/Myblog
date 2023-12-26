#include<stdio.h>
__global__ void hello_from_gpu()
{
   // 获取block的id
   const int bid = blockIdx.x;
   // 获取thread的id
   const int tid = threadIdx.x;
   // 获取y轴的id
   const int yid = threadIdx.y;
   // 打印出当前block和thread的id
   printf("hello word from block %d and thread (%d,%d)\n",bid,tid,yid);
}
int main()
{
   // 定义block的size
   const dim3 block_size(2,4);
   // 调用hello_from_gpu函数
   hello_from_gpu<<<1,block_size>>>();
   // 等待cuda计算完成
   cudaDeviceSynchronize();
   // 打印出helloword
   printf("helloword\n");
   return 0;
}
