
// OpenGL Graphics includes
#pragma comment(lib, "glew32.lib")
#include <helper_gl.h>
#include<Windows.h>
#include<fstream>
#if defined (__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif
#include<iostream>
#include<string>
// CUDA Runtime, Interop, and includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_profiler_api.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// CUDA utilities
#include <helper_cuda.h>

// Helper functions
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>
using namespace std;
typedef unsigned int uint;
typedef unsigned char uchar;

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
	"volume.ppm",
	NULL
};

const char *sReference[] =
{
	"ref_volume.ppm",
	NULL
};
const char *sSDKsample = "CUDA 3D Volume Render";

const char *volumeFilename = "ctneck_tumorcut_8bits.raw";
cudaExtent volumeSize = make_cudaExtent(512,512,512);

typedef unsigned char VolumeType;

uint width = 512, height = 512;
dim3 blockSize(16, 16);
dim3 gridSize;
float3 viewRotation=make_float3(-90.0,0.0,0.0);
float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
float invViewMatrix[12];//内参矩阵
float density = 0.36f;//密度
float brightness = 1.0f;//亮度
bool linearFiltering = true;

GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

StopWatchInterface *timer = 0;

// Auto-Verification Code
const int frameCheckNumber = 2;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
unsigned int frameCount = 0;

int *pArgc;
char **pArgv;
//新加
int frameNumber=16;
int colorKind=4;
int sampleValueRange=256;
float**prePms;
float**pms;
int**nms;
int**LUT;
int*LUT1D;
int cnt = 0;
int frameNumberNow = 0;
#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

extern "C" void setTextureFilterMode(bool bLinearFilter);
extern "C" void initCuda(void *h_volume, cudaExtent volumeSize,int max);
extern "C" void freeCudaBuffers();
extern "C" void render_kernel(size_t sizeLUT,int*a,dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,float density, float brightness);
extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);
//extern "C" void copyLUT(int *a, size_t sizeofLUT);
void adjustLUT(int *a)//对材料选择LUT进行符合材料同步模式的调整
{
	for (int j = 0; j < sampleValueRange; j++)
	{
		int minDistance = frameNumber;
		int minIndex=-1;
		int *temp = new int[frameNumber];
		int d = (frameNumber / colorKind);//公共基础分配范围
		int *indexM = new int[colorKind];//各材料的中心位置
		int *cntColor = new int[colorKind];//统计各种颜色的帧数
		for (int i = 0; i < colorKind; i++)
		{
			cntColor[i] = 0;
			indexM[i] = d*i + d / 2;//材料中心位置
		}
		for (int i = 0; i < frameNumber; i++)
		{
			temp[i] = LUT1D[i*sampleValueRange + j];//将LUT1D里用到的数据提取出来进行调整
			cntColor[temp[i]]++;//计算各材料帧数
			temp[i] = -1;//清空当前表
		}
		for (int i = 0; i < frameNumber; i++)
		{
			for (int k = 0; k < colorKind; k++)
			{
				if (cntColor[k]>0)
				{
					if (minDistance>abs(i - indexM[k]))//计算距离此处最近的中心位置，每一个中心位置代表不同的颜色
					{
						minDistance = abs(i - indexM[k]);
						minIndex = k;
					}
					else if (minDistance == abs(i - indexM[k]))//如果有两个中心位置距离相等，选择具体大的
					{
						if (indexM[k]>indexM[minIndex])
							minIndex = k;
					}
				}
			}
			cntColor[minIndex]--;
			temp[i] = minIndex;
		}
		for (int i = 0; i < frameNumber; i++)
		{
			LUT1D[i*sampleValueRange + j] = temp[i];//将调整后的数据重新写入材料选择表
		}
		free(temp);//释放内存
		free(indexM);
		free(cntColor);
	}
}
void initPixelBuffer();
int argmax(int sampleValue)
{
	int maxIndex = 0;
	for (int i = 0; i < colorKind; i++)
	{
		if (nms[i][sampleValue] > nms[maxIndex][sampleValue])
			maxIndex = i;
	}
	return maxIndex;
}
void getLut()
{
	LUT1D = new int[sampleValueRange*frameNumber];
	//分配内存
	prePms = new float*[colorKind];
	pms = new float*[colorKind];
	nms = new int*[colorKind];
	for (int i = 0; i < colorKind; i++)
	{
		prePms[i] = new float[sampleValueRange];
		pms[i] = new float[sampleValueRange];
		nms[i] = new int[sampleValueRange];
	}
	for (int i = 0; i < 122; i++)
	{
		for (int j = 1; j < colorKind; j++)
			prePms[j][i] = 0;
	}
	for (int i = 122; i < 124; i++)
	{
		prePms[1][i] = (float)(i - 122) / (float)2;
		prePms[2][i] = 0;
		prePms[3][i] = 0;
	}
	for (int i = 124; i < 256; i++)
	{
		//肿瘤
		if (i < 128)
			prePms[1][i] = 1.0;
		if (i >= 128 && i < 135)
			prePms[1][i] = (float)(135 - i) / (float)7;
		if (i >= 135)
			prePms[1][i] = 0.0;
		//血管
		if (i < 125)
			prePms[2][i] = 0.0;
		if (i >= 125 && i < 132)
			prePms[2][i] = (float)(i - 125) / (float)7;
		if (i >= 132 && i < 140)
			prePms[2][i] = 1;
		if (i >= 140 && i < 142)
			prePms[2][i] = (float)(142 - i) / (float)2;
		if (i >= 142)
			prePms[2][i] = 0.0;
		//骨头
		if (i < 133)
			prePms[3][i] = 0;
		if (i >= 133 && i < 140)
			prePms[3][i] = (float)(i - 133) / (float)7;
		if (i >= 140)
			prePms[3][i] = 1;
	}
	for (int i = 0; i < 256; i++)
	{
		prePms[0][i] = 1;
		for (int j = 1; j < colorKind; j++)
		{
			prePms[0][i] -= prePms[j][i];
		}
		if (prePms[0][i] < 0)
			prePms[0][i] = 0;
	}
	ofstream output("E:\lut.txt");
/*	for (int i = 0; i < 4; i++)
	{
	for (int j = 0; j < 256; j++)
	{
	output << pms[i][j] << " ";
	}
	output << endl;
	}
	*/
	float*sum = new float[sampleValueRange];
	for (int i = 0; i <sampleValueRange; i++)
	{
		sum[i] = 0;
		for (int j = 0; j < colorKind; j++)
			sum[i] += prePms[j][i];
//		cout << sum[i] << endl;
		for (int j = 0; j < colorKind; j++)
		{
//			cout << sum[i] << endl;
			pms[j][i] = prePms[j][i]/sum[i];
			float s = pms[j][i] * (float)frameNumber;
			nms[j][i] = (int)s;
//			cout << nms[j][i] << " ";
		}
//		cout << endl;
	}

	int m;
	for (int i = 0; i < sampleValueRange; i++)
	{
		for (int j = 0; j < frameNumber; j++)
		{
			m = argmax(i);
//			cout << m << " ";
			LUT1D[j*sampleValueRange + i] = m;
			nms[m][i]--;
		}
//		cout << endl;
	}
	adjustLUT(LUT1D);
//	copyLUT(LUT1D, sampleValueRange*frameNumber*sizeof(int));
	for (int i = 0; i< sampleValueRange*frameNumber;i++)
	{
		if (i != 0 && i % 256 == 0)
			output << endl;
		output << LUT1D[i] << " ";
		
	}

}
void computeFPS()
{
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		char fps[256];
		float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		sprintf(fps, "Volume Render: %3.1f fps", ifps);

		glutSetWindowTitle(fps);
		fpsCount = 0;

		fpsLimit = (int)MAX(1.f, ifps);
		sdkResetTimer(&timer);
	}
}

// render image using CUDA
void render()
{
	size_t sizeofLUT = (2 << 11)*sizeof(int);
	copyInvViewMatrix(invViewMatrix, sizeof(float4) * 3);//把相机内参矩阵从host传递到device
	//pbo=像素缓冲区对象
	// map PBO to get CUDA device pointer
	uint *d_output;
	// map PBO to get CUDA device pointer
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,cuda_pbo_resource));
	//printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);
	// clear image
	checkCudaErrors(cudaMemset(d_output, 0, width*height * 4));
	//做好像素缓冲区即PBO的预备工作
	// call CUDA kernel, writing results to PBO
	render_kernel(sizeofLUT,LUT1D,gridSize, blockSize, d_output, width, height, density, brightness);

	getLastCudaError("kernel failed");
	//解除映射
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

// display results using OpenGL (called by GLUT)
void display()
{
	sdkStartTimer(&timer);

	// use OpenGL to build view matrix
	GLfloat modelView[16];
	glMatrixMode(GL_MODELVIEW);//创建模型视图，GL_PROJECTION 投影, GL_MODELVIEW 模型视图, GL_TEXTURE 纹理。
	glPushMatrix();//保存当前位置
	glLoadIdentity();// glLoadIdentity()该函数的功能是重置当前指定的矩阵为单位矩阵.在语义上，其等同于用单位矩阵调用glLoadMatrix()。
	/* 
	单位矩阵就是对角线上都是1，其余元素皆为0的矩阵。

      当您调用glLoadIdentity()之后，您实际上将当前点移到了屏幕中心：类似于一个复位操作
      1.X坐标轴从左至右，Y坐标轴从下至上，Z坐标轴从里至外。
      2.OpenGL屏幕中心的坐标值是X和Y轴上的0.0f点。
      3.中心左面的坐标值是负值，右面是正值。
         移向屏幕顶端是正值，移向屏幕底端是负值。
         移入屏幕深处是负值，移出屏幕则是正值。
	*/
	glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);//绕x轴旋转
	glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);//绕y轴旋转
	/*
	void glRotatef(GLfloat angle,  GLfloat x,  GLfloat y,  GLfloat z);
      其中,angle为旋转的角度,单位为度.接下来的xyz分别表示xyz轴，要绕着某轴旋转，只要将它的值置为1即可。
	 */
	glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);//移动点，(x,y,z)->(x+a,y+b,z+c)
	glGetFloatv(GL_MODELVIEW_MATRIX, modelView);//获得当前矩阵数据
	glPopMatrix();//回复原来位置
	//单位矩阵旋转后即为旋转矩阵
	//矩阵转置
	invViewMatrix[0] = modelView[0];
	invViewMatrix[1] = modelView[4];
	invViewMatrix[2] = modelView[8];
	invViewMatrix[3] = modelView[12];
	invViewMatrix[4] = modelView[1];
	invViewMatrix[5] = modelView[5];
	invViewMatrix[6] = modelView[9];
	invViewMatrix[7] = modelView[13];
	invViewMatrix[8] = modelView[2];
	invViewMatrix[9] = modelView[6];
	invViewMatrix[10] = modelView[10];
	invViewMatrix[11] = modelView[14];

	render();

	// display results
	glClear(GL_COLOR_BUFFER_BIT);

	// draw image from PBO
	glDisable(GL_DEPTH_TEST);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
#if 0
	// draw using glDrawPixels (slower)
	glRasterPos2i(0, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
#else
	// draw using texture

	// copy from pbo to texture
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	// draw textured quad
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0, 0);
	glVertex2f(0, 0);
	glTexCoord2f(1, 0);
	glVertex2f(1, 0);
	glTexCoord2f(1, 1);
	glVertex2f(1, 1);
	glTexCoord2f(0, 1);
	glVertex2f(0, 1);
	glEnd();

	glDisable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);
#endif

	glutSwapBuffers();
	glutReportErrors();

	sdkStopTimer(&timer);
	computeFPS();
	frameNumberNow++;
	cout <<"count:"<<cnt<< endl;
	cnt++;
	if (cnt % 16 == 0)//控制每播放1帧旋转30度
	{
		cnt = 0;
		viewRotation.y += (float)30;
		if (viewRotation.y >= 360)
			viewRotation.y = (int)viewRotation.y % 360;
	}
	glFlush();
	glutPostRedisplay();

	Sleep(330);//控制动画速度
}

void idle()
{
	glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 27:
#if defined (__APPLE__) || defined(MACOSX)
		exit(EXIT_SUCCESS);
#else
		glutDestroyWindow(glutGetWindow());
		return;
#endif
		break;

	case 'f':
		linearFiltering = !linearFiltering;
		setTextureFilterMode(linearFiltering);
		break;

	case '+':
		density += 0.01f;
		break;

	case '-':
		density -= 0.01f;
		break;

	case ']':
		brightness += 0.1f;
		break;

	case '[':
		brightness -= 0.1f;
		break;

	default:
		break;
	}

	printf("density = %.2f, brightness = %.2f\n", density, brightness);
	glutPostRedisplay();
}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
	/*
	if (state == GLUT_DOWN)
	{
		buttonState |= 1 << button;
	}
	else if (state == GLUT_UP)
	{
		buttonState = 0;
	}

	ox = x;
	oy = y;
	glutPostRedisplay();
	*/
}

void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - ox);
	dy = (float)(y - oy);

	if (buttonState == 4)
	{
		// right = zoom
		viewTranslation.z += dy / 100.0f;
	}
	else if (buttonState == 2)
	{
		// middle = translate
		viewTranslation.x += dx / 100.0f;
		viewTranslation.y -= dy / 100.0f;
	}
	else if (buttonState == 1)
	{
		// left = rotate
		viewRotation.x += dy / 5.0f;
		viewRotation.y += dx / 5.0f;
	}

	ox = x;
	oy = y;
	glutPostRedisplay();
}

int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

void reshape(int w, int h)
{
	width = w;
	height = h;
	initPixelBuffer();

	// calculate new grid size
	gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

	glViewport(0, 0, w, h);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void cleanup()
{
	sdkDeleteTimer(&timer);

	freeCudaBuffers();

	if (pbo)
	{
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffers(1, &pbo);
		glDeleteTextures(1, &tex);
	}
	// Calling cudaProfilerStop causes all profile data to be
	// flushed before the application exits
	checkCudaErrors(cudaProfilerStop());
}

void initGL(int *argc, char **argv)
{
	// initialize GLUT callback functions
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);//双通道RGB
	glutInitWindowSize(width, height);
	glutCreateWindow("CUDA volume rendering");

	if (!isGLVersionSupported(2, 0) ||
		!areGLExtensionsSupported("GL_ARB_pixel_buffer_object"))
	{
		printf("Required OpenGL extensions are missing.");
		exit(EXIT_SUCCESS);
	}
}

void initPixelBuffer()
{
	if (pbo)
	{
		// unregister this buffer object from CUDA C
		checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

		// delete old buffer
		glDeleteBuffers(1, &pbo);
		glDeleteTextures(1, &tex);
	}

	// create pixel buffer object for display
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW_ARB);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

	// create texture for display
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
}

// Load raw data from disk
void *loadRawFile(char *filename, size_t size)
{
	FILE *fp = fopen(filename, "rb");

	if (!fp)
	{
		fprintf(stderr, "Error opening file '%s'\n", filename);
		return 0;
	}

	void *data = malloc(size);
	size_t read = fread(data, 1, size, fp);
	fclose(fp);

#if defined(_MSC_VER_)
	printf("Read '%s', %Iu bytes\n", filename, read);
#else
	printf("Read '%s', %zu bytes\n", filename, read);
#endif

	return data;
}

void runSingleTest(const char *ref_file, const char *exec_path)
{
	bool bTestResult = true;

	uint *d_output;
	checkCudaErrors(cudaMalloc((void **)&d_output, width*height*sizeof(uint)));
	checkCudaErrors(cudaMemset(d_output, 0, width*height*sizeof(uint)));

	float modelView[16] =
	{
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 4.0f, 1.0f
	};

	invViewMatrix[0] = modelView[0];
	invViewMatrix[1] = modelView[4];
	invViewMatrix[2] = modelView[8];
	invViewMatrix[3] = modelView[12];
	invViewMatrix[4] = modelView[1];
	invViewMatrix[5] = modelView[5];
	invViewMatrix[6] = modelView[9];
	invViewMatrix[7] = modelView[13];
	invViewMatrix[8] = modelView[2];
	invViewMatrix[9] = modelView[6];
	invViewMatrix[10] = modelView[10];
	invViewMatrix[11] = modelView[14];

	// call CUDA kernel, writing results to PBO
	copyInvViewMatrix(invViewMatrix, sizeof(float4) * 3);

	// Start timer 0 and process n loops on the GPU
	int nIter = 10;

	for (int i = -1; i < nIter; i++)
	{
		if (i == 0)
		{
			cudaDeviceSynchronize();
			sdkStartTimer(&timer);
		}

		render_kernel((2 << 26)*sizeof(int),LUT1D, gridSize, blockSize, d_output, width, height, density, brightness);
	}

	cudaDeviceSynchronize();
	sdkStopTimer(&timer);
	// Get elapsed time and throughput, then log to sample and master logs
	double dAvgTime = sdkGetTimerValue(&timer) / (nIter * 1000.0);
	printf("volumeRender, Throughput = %.4f MTexels/s, Time = %.5f s, Size = %u Texels, NumDevsUsed = %u, Workgroup = %u\n",
		(1.0e-6 * width * height) / dAvgTime, dAvgTime, (width * height), 1, blockSize.x * blockSize.y);


	getLastCudaError("Error: render_kernel() execution FAILED");
	checkCudaErrors(cudaDeviceSynchronize());

	unsigned char *h_output = (unsigned char *)malloc(width*height * 4);
	checkCudaErrors(cudaMemcpy(h_output, d_output, width*height * 4, cudaMemcpyDeviceToHost));

	sdkSavePPM4ub("volume.ppm", h_output, width, height);
	bTestResult = sdkComparePPM("volume.ppm", sdkFindFilePath(ref_file, exec_path), MAX_EPSILON_ERROR, THRESHOLD, true);

	cudaFree(d_output);
	free(h_output);
	cleanup();

	exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	pArgc = &argc;
	pArgv = argv;
	getLut();
	char *ref_file = NULL;

#if defined(__linux__)
	setenv("DISPLAY", ":0", 0);
#endif

	//start logs
	printf("%s Starting...\n\n", sSDKsample);
	// First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	initGL(&argc, argv);

	findCudaDevice(argc, (const char **)argv);

	// parse arguments
	char *filename;

	int n;

	// load volume data
	char *path = sdkFindFilePath(volumeFilename, argv[0]);

	if (path == 0)
	{
		printf("Error finding file '%s'\n", volumeFilename);
		exit(EXIT_FAILURE);
	}

	size_t size = volumeSize.width*volumeSize.height*volumeSize.depth*sizeof(VolumeType);
	void *h_volume = loadRawFile(path, size);
	int max = 0;
	uchar*dv = (uchar*)h_volume;
	for (int i = 0; i < volumeSize.depth*volumeSize.height*volumeSize.width; i++)
	{
		if (max < dv[i])
		{
			max = dv[i];
//			cout << max << endl;
		}
	}
	initCuda(h_volume, volumeSize,max);//创建两个纹理，传递函数纹理以及原始数据纹理
	free(h_volume);

	sdkCreateTimer(&timer);

	printf("Press '+' and '-' to change density (0.01 increments)\n"
		"      ']' and '[' to change brightness\n");

	// calculate new grid size
	gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

	if (ref_file)
	{
		cout << "ref_file"<<endl;
		runSingleTest(ref_file, argv[0]);
	}
	else
	{
		// This is the normal rendering path for VolumeRender
		glutDisplayFunc(display);
		glutKeyboardFunc(keyboard);
		glutMouseFunc(mouse);
		glutMotionFunc(motion);
		glutReshapeFunc(reshape);
		glutIdleFunc(idle);

		initPixelBuffer();

#if defined (__APPLE__) || defined(MACOSX)
		atexit(cleanup);
#else
		glutCloseFunc(cleanup);
#endif
		
		glutMainLoop();
	}
}
