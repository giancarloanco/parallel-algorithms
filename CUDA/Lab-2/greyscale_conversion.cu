#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdio>

#define CHANNELS 3

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
    
    f = fopen("Greyscale.bmp","wb");

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

// we have 3 channels corresponding to RGB
// The input image is encoded as unsigned characters [0, 255]
__global__
void colorToGreyscaleConversion(unsigned char * Pout, unsigned char * Pin, int width, int height)
{
	int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;

    if (Col < width && Row < height)
    {
		// get 1D coordinate for the grayscale image
		int greyOffset = Row*width + Col;
		// one can think of the RGB image having
		// CHANNEL times columns than the grayscale image
		int rgbOffset = greyOffset*CHANNELS;
		unsigned char r = Pin[rgbOffset ]; // red value for pixel
		unsigned char g = Pin[rgbOffset + 2]; // green value for pixel
		unsigned char b = Pin[rgbOffset + 3]; // blue value for pixel
		// perform the rescaling and store it
		// We multiply by floating point constants
		Pout[greyOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
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
	colorToGreyscaleConversion<<<dimGrid, dimBlock>>>(Img_Out_Device, Img_In_Device, width, height);
	cudaMemcpy(Img_Out_Host, Img_Out_Device, size, cudaMemcpyDeviceToHost);
	
	writeBMP(Img_Out_Host, width, height);
    return 0;
}
