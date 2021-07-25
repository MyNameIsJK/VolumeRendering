#ifndef _VOLUMERENDER_KERNEL_CU_
#define _VOLUMERENDER_KERNEL_CU_

#include <helper_cuda.h>
#include <helper_math.h>

typedef unsigned int  uint;
typedef unsigned char uchar;

cudaArray *d_volumeArray = 0;
cudaArray *d_transferFuncArray;

typedef unsigned char VolumeType;//typedef unsigned short VolumeType;

cudaTextureObject_t	texObject; // For 3D texture
cudaTextureObject_t transferTex; // For 1D transfer function texture
cudaTextureObject_t texLUT;

typedef struct
{
	float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix
__device__ int maxSampleValue=255;//����ǿ��������ֵ��cuda�����׼��Ϊ��0��1������ÿ������ֵ�������ֵ���˴����ڱ�׼����Ļ�ԭ
int framePre=0;//��ǰ֡
int*LUTD;//render��ʱ��������ϵ�LUT�������豸��
__constant__ float4 tf[4]=
{
	{ 0.0, 0.0, 0.0, 0.0, },
	{ 0.0, 1.0, 0.0, 0.05, },
	{ 1.0, 0.0, 0.0, 0.15, },
	{ 1.0, 1.0, 1.0, 0.1 },
};//���ֲ��ϵ�RGBֵ�Լ�͸����
struct Ray
{
	float3 o;   // origin
	float3 d;   // direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
	//make_float3(1.0)=(1,1,1)
	//�˴�/����Ϊ��a/b,c/d,e/f��
	// compute intersection of ray with all six bbox planes
	float3 invR = make_float3(1.0f) / r.d;
	//�˷�Ϊ��λ��
	float3 tbot = invR * (boxmin - r.o);
	float3 ttop = invR * (boxmax - r.o);

	// re-order intersections to find smallest and largest on each axis
	float3 tmin = fminf(ttop, tbot);
	float3 tmax = fmaxf(ttop, tbot);

	// find the largest tmin and the smallest tmax
	float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
	float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
	float3 r;
	r.x = dot(v, make_float3(M.m[0]));
	r.y = dot(v, make_float3(M.m[1]));
	r.z = dot(v, make_float3(M.m[2]));
	return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
	float4 r;
	r.x = dot(v, M.m[0]);//�������
	r.y = dot(v, M.m[1]);
	r.z = dot(v, M.m[2]);
	r.w = 1.0f;
	return r;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
	rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
	rgba.y = __saturatef(rgba.y);
	rgba.z = __saturatef(rgba.z);
	rgba.w = __saturatef(rgba.w);
	return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) | (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
}

//�ӵ�λ�ó�ʼΪ��0��0��4����ƽ��λ��Ϊ��0��0��2����ƽ���СΪ2*2
__global__ void d_render(int frame,int*a,uint *d_output, uint imageW, uint imageH,float density, float brightness, cudaTextureObject_t	tex,cudaTextureObject_t	transferTex)
{
	//ǰ�ò���
	const int maxSteps = 500;
	const float tstep = 0.01f;//�������
	const float opacityThreshold = 0.95f;
	const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
	const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);
	//boxMin��boxMax����volume���������㣬��ȷ��volume�������Ǵ�boxMin��boxMax
	//�߳�����
	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;
	//ȷ���̵߳�λ��ͼ����
	if ((x >= imageW) || (y >= imageH)) return;
	//x,yΪ����ͼ���е����꣬Ҫת������ƽ���С�-1��1�������������²���
	float u = (x / (float)imageW)*2.0f - 1.0f;
	float v = (y / (float)imageH)*2.0f - 1.0f;
	// calculate eye ray in world space
	Ray eyeRay;
	//ÿһ�λ�����display�ж��Ḵλ��Ҳ����˵ÿ����ת�ļ��㶼�����ӵ�λ��Ϊ��0��0��4��������ת
	eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));//�ӵ�λ�ã����ڲξ����е�ƽ�ƾ���
	eyeRay.d = normalize(make_float3(u, v, -2.0f));
	eyeRay.d = mul(c_invViewMatrix, eyeRay.d);//���߷���
	// find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

	if (!hit) return;

	if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

	// march along ray from front to back, accumulating color
	float4 sum = make_float4(0.0f);
	float t = tnear;
	float3 pos = eyeRay.o + eyeRay.d*tnear;//������volume�Ľ��㣬�����
	float3 step = eyeRay.d*tstep;//������һ��������ľ���

	for (int i = 0; i < maxSteps; i++)
	{
		// read from 3D texture
		// remap position to [0, 1] coordinates
		float sample = tex3D<float>(tex, pos.x*0.5f + 0.5f, pos.y*0.5f + 0.5f, pos.z*0.5f + 0.5f);//��Ϊ�������е������ǡ�0��1���ģ�������Ҫ����ӳ��
		//sample *= 64.0f;    // scale for 10-bit data
//		int indexL = 256;
		float indexLF = sample*maxSampleValue;
		int indexL = frame*256+indexLF;
//		int index = 3;
		int index = a[indexL];
		// lookup in transfer function texture
		//colΪtransferfuc��һ��
//		index = index % 3;
		float4 col = tf[index];
//		float4 col =tex1D<float4>(transferTex, sample);
//		printf("%f %f %f %f\n",col.x,col.y,col.z,col.w);
//		float4 col = tex1D<float4>(transferTex, index);//������ֵ���е���
		col.w *= density;

		// "under" operator for back-to-front blending
		//sum = lerp(sum, col, col.w);

		// pre-multiply alpha
		col.x *= col.w;
		col.y *= col.w;
		col.z *= col.w;
		// "over" operator for front-to-back blending
		sum = sum + col*(1.0f - sum.w);

		// exit early if opaque
		if (sum.w > opacityThreshold)
			break;

		t += tstep;

		if (t > tfar) break;

		pos += step;
	}

	sum *= brightness;

	// write output color
	d_output[y*imageW + x] = rgbaFloatToInt(sum);
}

extern "C"
void setTextureFilterMode(bool bLinearFilter)
{
	if (texObject)
	{
		checkCudaErrors(cudaDestroyTextureObject(texObject));
	}
	cudaResourceDesc            texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_volumeArray;

	cudaTextureDesc             texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = true;
	texDescr.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;

	texDescr.addressMode[0] = cudaAddressModeWrap;
	texDescr.addressMode[1] = cudaAddressModeWrap;
	texDescr.addressMode[2] = cudaAddressModeWrap;

	texDescr.readMode = cudaReadModeElementType;

	checkCudaErrors(cudaCreateTextureObject(&texObject, &texRes, &texDescr, NULL));

}

extern "C"
void initCuda(void *h_volume, cudaExtent volumeSize,int max)
{
	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
	checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));
	maxSampleValue = max;
	// copy data to 3D array
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr(h_volume, volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
	copyParams.dstArray = d_volumeArray;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams));

	cudaResourceDesc            texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_volumeArray;

	cudaTextureDesc             texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = true; // access with normalized texture coordinates
	texDescr.filterMode = cudaFilterModeLinear; // linear interpolation

	texDescr.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;

	texDescr.readMode = cudaReadModeNormalizedFloat;

	checkCudaErrors(cudaCreateTextureObject(&texObject, &texRes, &texDescr, NULL));

	// create transfer function texture
/*	float4 transferFunc[] =
	{
		{ 0.0, 0.0, 0.0, 0.0, },
		{ 0.0, 0.0, 0.0, 0.0, },
		{ 1.0, 0.5, 0.0, 1.0, },
		{ 1.0, 1.0, 0.0, 1.0, },
		{ 0.0, 1.0, 0.0, 1.0, },
		{ 0.0, 1.0, 1.0, 1.0, },
		{ 0.0, 0.0, 1.0, 1.0, },
		{ 1.0, 0.0, 1.0, 1.0, },
		{ 0.0, 0.0, 0.0, 0.0, },
	};

	cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
	cudaArray *d_transferFuncArray;
	//�Ѵ��ݺ������ڴ濽�����豸
	checkCudaErrors(cudaMallocArray(&d_transferFuncArray, &channelDesc2, sizeof(transferFunc) / sizeof(float4), 1));
	checkCudaErrors(cudaMemcpyToArray(d_transferFuncArray, 0, 0, transferFunc, sizeof(transferFunc), cudaMemcpyHostToDevice));

	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_transferFuncArray;


	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = true; // access with normalized texture coordinates
	texDescr.filterMode = cudaFilterModeLinear;

	texDescr.addressMode[0] = cudaAddressModeClamp; // wrap texture coordinates

	texDescr.readMode = cudaReadModeElementType;

	checkCudaErrors(cudaCreateTextureObject(&transferTex, &texRes, &texDescr, NULL));
	*/
}

extern "C"
void freeCudaBuffers()
{
	checkCudaErrors(cudaDestroyTextureObject(texObject));
	checkCudaErrors(cudaDestroyTextureObject(transferTex));
	checkCudaErrors(cudaFreeArray(d_volumeArray));
	checkCudaErrors(cudaFreeArray(d_transferFuncArray));
}


extern "C"
void render_kernel(size_t sizeLUT,int*a,dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,float density, float brightness)//��ת����d_render����
{
//	for (int i = 0; i < 16; i++)
//		printf("%d ", a[i]);
//	printf("\n");
	checkCudaErrors(cudaMalloc((void**)&LUTD,sizeLUT));
	checkCudaErrors(cudaMemcpy(LUTD, a, sizeLUT, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(a, LUTD, sizeLUT, cudaMemcpyDeviceToHost));
//	for (int i = 0; i < 16; i++)
//		printf("%d ", a[i]);
	d_render << <gridSize, blockSize >> >(framePre,LUTD,d_output, imageW, imageH, density,
		brightness,texObject, transferTex);
	printf("%d\n", framePre);
	framePre++;//������LUT�л�ò�������
	if (framePre >= 16)
		framePre = framePre % 16;
}

extern "C"
void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
	checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));//��ʼ��__device__����
}


#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
