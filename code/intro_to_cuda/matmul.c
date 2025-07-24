#include <stdio.h>

void matmul(float*A,float*B,float*C,int Width)
{
    for (int i = 0; i < Width;++i)
    {
        for (int j = 0; j< Width;++j)
        {
            float sum = 0;
            for (int k = 0;k<Width;++k) // innermost loop that iterates over the variable k and steps through one row of matrix A and one column of matrix B
            {
                float a = A[i*Width+k];//accessing matrix A
                float b = B[k*Width+j];
                sum += a *b;
            }
            C[i*Width+j]=sum;
        }
    }
        

}