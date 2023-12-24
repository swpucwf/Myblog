#include<stdio.h>
__global__ void hello_from_gpu()
{
//  线程位置如何索引；
/*gridDim.x ：该变量的数值等与执行配置中变量grid_size的数值。
blockDim.x: 该变量的数值等与执行配置中变量block_size的数值。
在核函数中预定义了如下标识线程的内建变量：
blockIdx.x :该变量指定一个线程在一个网格中的线程块指标。其取值范围是从0到gridDim.x-1
threadIdx.x：该变量指定一个线程在一个线程块中的线程指标，其取值范围是从0到blockDim.x-1
*/
   const int bid = blockIdx.x; // block
   const int tid = threadIdx.x;// thread
   const int blockdim = gridDim.x; // the number of block;
   const int threaddim = blockDim.x; // the number of thread;
   printf("hello word from blockdim %d and threaddim %d\n",blockdim,threaddim);
   printf("hello word from block %d and thread %d\n",bid,tid);
}
int main()
{
   hello_from_gpu<<<2,4>>>();
  

   cudaDeviceSynchronize(); 
   printf("helloword\n");
   return 0;
}
