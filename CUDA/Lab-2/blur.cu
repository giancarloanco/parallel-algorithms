#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdio>

#define BLUR_SIZE 1

using namespace std;

unsigned char* readBMP(char* filename, int &my_width, int &my_height)
{
    FILE* f = fopen(filename, "rb");
    unsigned char info[54];
    fread(info, sizeof(unsigned char), 54, f); 

    int width = *(int*)&info[18];
    int height = *(int*)&info[22];

    int size = 3 * width * height;
    unsigned char* data = new unsigned char[size]; 
    fread(data, sizeof(unsigned char), size, f); 
    fclose(f);
    my_width = width;
	my_height = height;

    return data;
}

void writeBMP(unsigned char* img, int w, int h)
{
    FILE *f;
    int filesize = 54 + 3*w*h;  //w is your image width, h is image height, both int
    unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
    unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
    unsigned char bmppad[3] = {0,0,0};
    bmpfileheader[ 2] = (unsigned char)(filesize    );
    bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
    bmpfileheader[ 4] = (unsigned char)(filesize>>16);
    bmpfileheader[ 5] = (unsigned char)(filesize>>24);
    bmpinfoheader[ 4] = (unsigned char)(       w    );
    bmpinfoheader[ 5] = (unsigned char)(       w>> 8);
    bmpinfoheader[ 6] = (unsigned char)(       w>>16);
    bmpinfoheader[ 7] = (unsigned char)(       w>>24);
    bmpinfoheader[ 8] = (unsigned char)(       h    );
    bmpinfoheader[ 9] = (unsigned char)(       h>> 8);
    bmpinfoheader[10] = (unsigned char)(       h>>16);
    bmpinfoheader[11] = (unsigned char)(       h>>24);
    
    f = fopen("Blur.bmp","wb");

    fwrite(bmpfileheader,1,14,f);
    fwrite(bmpinfoheader,1,40,f);
    //for(int i=0; i<h; i++)
    for(int i=h-1; i>=0; i--)
    {
        fwrite(img+(w*(h-i-1)*3),3,w,f);
        fwrite(bmppad,1,(4-(w*3)%4)%4,f);
    }
    free(img);
    fclose(f);
}

__global__
void blurKernel(unsigned char *in, unsigned char *out, int w, int h)
{
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(Col<w && Row<h)
    {
        int pixVal = 0;
        int pixels = 0;
        
        for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow)
        {
            for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol)
            {
                int curRow = Row + blurRow;
                int curCol = Col + blurCol;
                
                if(curRow>-1 && curRow<h && curCol>-1 && curCol<w)
                {
                    pixVal += in[curRow * w +curCol];
                    pixels++;
                }
            }
            out[Row * w + Col] = (unsigned char)(pixVal/pixels);
        }
    }
       
}

int main()
{
    unsigned char* Img_In_Host;
    unsigned char* Img_Out_Host;
    unsigned char* Img_In_Device;
    unsigned char* Img_Out_Device;
    int width = 0;
    int height = 0;
    
    Img_In_Host = readBMP("McLaren.bmp", width, height);
    
    int size = width * height * sizeof(unsigned char);
    
    Img_Out_Host = (unsigned char*)malloc(size * sizeof(unsigned char));

    cudaMalloc((void **) &Img_In_Device, size*3);
    cudaMemcpy(Img_In_Device, Img_In_Host, size*3, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &Img_Out_Device, size);
    cudaMemcpy(Img_Out_Device, Img_Out_Host, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(width/16.0), ceil(height/16.0), 1);
    dim3 dimBlock(16, 16, 1);
    blurKernel<<<dimGrid, dimBlock>>>(Img_In_Device, Img_Out_Device, width, height);
    cudaMemcpy(Img_Out_Host, Img_Out_Device, size, cudaMemcpyDeviceToHost);
    
    writeBMP(Img_Out_Host, width, height);
    return 0;
}
