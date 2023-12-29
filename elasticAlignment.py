import ctypes
import numpy as np
from cuda import cuda, cudart
from common import common
from common.helper_cuda import checkCudaErrors, findCudaDeviceDRV
import trsfile
from trsfile import trs_open, Trace, SampleCoding, TracePadding, Header
from trsfile.parametermap import TraceParameterMap, TraceParameterDefinitionMap
from trsfile.traceparameter import ByteArrayParameter, ParameterType, TraceParameterDefinition
import os
import time

a = time.time()
traceSet = []
amountOfTraces = 0
with trsfile.open('BenchmarkTraces/200traces2k.trs', 'r') as traces:
    # Show all headers
    for header, value in traces.get_headers().items():
        print(header, '=', value)

    for i, trace in enumerate(traces):
        traceSet = np.concatenate((traceSet, trace))
        amountOfTraces += 1
    referenceTrace = traces[0]

fastdtwDrv = '''\
#define INT_MAX 2147483647
#define DBL_MAX 340282346638528859811704183484516925440.000000
__device__ float distance(float x, float y) {
    return (x - y) * (x - y);
}

__device__ void linear_to_2d_index(int linear_index, int num_rows, int num_cols, int *x, int *y) 
{
    *x = ((linear_index - 1) % num_cols)+1;
    *y = ((linear_index - 1) / num_cols)+1;
}

__device__ int index_2d_to_linear(int x, int y, int num_cols) {
        return (y) * num_cols + x;
    }

struct Range {
    int start;
    int end;
};

__device__ void reverseArray(int arr[], int size) {
    int start = 0;
    int end = size - 1;
    while (start < end) {
        int temp = arr[start];
        arr[start] = arr[end];
        arr[end] = temp;

        start++;
        end--;
    }
}

__device__ int warppath(int* windowArray, int originalN, int lenX, float* global, int* C) {
	int threadID =((blockIdx.x + gridDim.x * blockIdx.y) * (blockDim.x * blockDim.y)) + threadIdx.x + blockDim.x * threadIdx.y;
    int n = lenX*lenX;
    C[(originalN*2*threadID)] = n;
    int warpPathLength = 0;
    float l1, l2, l3;
    int nextx, nexty;
    float smallest;
    int x;
    int y;
    int lowerX;
    int upperX;
    int leftlowerX;
    int leftupperX;

    while (n != 1) {
        linear_to_2d_index(n, lenX, lenX, &x, &y);
        lowerX = windowArray[(y - 1)*2+(originalN*2*threadID)];
        upperX = windowArray[(y - 1)*2+1+originalN*2*threadID];


        if (y > 1) {
            leftlowerX = windowArray[(y - 2)*2+(originalN*2*threadID)];
            leftupperX = windowArray[(y - 2)*2+1+(originalN*2*threadID)];
            if ((leftlowerX <= x) && (x <= leftupperX)) {
                l1 = global[x-leftlowerX+((y-2)*lenX)+(originalN*originalN*threadID)];
            } else {
                l1 = DBL_MAX; 
            }

            if ((leftlowerX <= x - 1) && (x - 1 <= leftupperX)) {
                l2 = global[x-1-leftlowerX+((y-2)*lenX)+(originalN*originalN*threadID)];
            } else {
                l2 = DBL_MAX;
            }
        } else {
            l1 = DBL_MAX;
            l2 = DBL_MAX;
        }

        if (x == lowerX) {
            l3 = DBL_MAX;
        } else {
            l3 = global[x-1-lowerX+((y-1)*lenX)+(originalN*originalN*threadID)];
        }

        if (l1 < l2) {
            nextx = x;
            nexty = y - 1;
            smallest = l1;
        } else {
            nextx = x - 1;
            nexty = y - 1;
            smallest = l2;
        }
        if (l3 < smallest) {
            nextx = x - 1;
            nexty = y;
        }
        n = index_2d_to_linear(nextx, nexty-1, lenX);
        warpPathLength++;
        C[(originalN*2*threadID)+warpPathLength] = n;
    }
    return (warpPathLength);
}

__device__ void defaultWindow(int* windowArray, int nX, int nY, int originalN) {
	int threadID =((blockIdx.x + gridDim.x * blockIdx.y) * (blockDim.x * blockDim.y)) + threadIdx.x + blockDim.x * threadIdx.y;
    for (int i = 0; i < 2*nY; ++i) {
        if(i % 2 == 0) {
            windowArray[i+(originalN*2*threadID)] = 1;
        }
        else {
            windowArray[i+(originalN*2*threadID)] = nX;
        }
    }
    return;
}

__device__ int dtw(float* X, float* Y, int* windowArray,const int lenX, const int lenY, float* global,int traceLowerResolutionIterator, int originalN, int* C) {
	int threadID =((blockIdx.x + gridDim.x * blockIdx.y) * (blockDim.x * blockDim.y)) + threadIdx.x + blockDim.x * threadIdx.y;
    int dimsArray[] = {lenX, lenY};
    for (int i = 0; i < lenX; ++i) {
        for(int j = 0; j < lenY; j++) {
            global[i*lenX+j+(originalN*originalN*threadID)] = DBL_MAX;
        }
    }

    int rightlowerX = 0;
    int rightupperX = 0;
	int lowerX = 0;
	int upperX = 0;
	int xoff = 0;
	float m = 0.0;
	float cur = 0.0;
    float prev = DBL_MAX;
    int x = lowerX;
    struct Range range1;
    struct Range range2;
    struct Range range3;
    
    for (int y = 1; y <= lenY; y++) {
        lowerX = windowArray[(y - 1)*2+(originalN*2*threadID)];
        upperX = windowArray[(y - 1)*2+1+(originalN*2*threadID)];
        float Yy = Y[y - 1+traceLowerResolutionIterator+(originalN*2*threadID)];

        if (y<lenY) {
            rightlowerX = windowArray[y*2+(originalN*2*threadID)];
            rightupperX = windowArray[y*2+1+(originalN*2*threadID)];
            range1.start = lowerX;
            range1.end = rightlowerX;
            int o1 = (rightlowerX > lowerX) ? rightlowerX : lowerX;
            int o2 = (rightupperX < upperX) ? rightupperX : upperX;
            range2.start = o1;
            range2.end = o2+1;
            range3.start = rightupperX +1;
            range3.end = upperX+1;
        }
        else {
            range1.start = lowerX;
            range1.end = upperX+1;
            range2.start = 0;
            range2.end = 0;
            range3.start = 0;
            range3.end = 0;

        }
        cur = 0.0;
        prev = DBL_MAX;

        for (int x = range1.start;x<range1.end;x++) {
            xoff = x-lowerX+1;
            m = global[(y-1)*lenX+xoff-1+(originalN*originalN*threadID)];
            m = (m < prev) ? m : prev;
            m = (m == DBL_MAX) ? 0.0 : m;
            cur = distance(X[x-1+traceLowerResolutionIterator+(originalN*2*threadID)], Yy) + m;
            global[(y-1)*lenX+xoff-1+(originalN*originalN*threadID)] = cur;
            prev = cur;
        }

        for (int x = range2.start;x<range2.end;x++) {
            xoff = x-lowerX+1;
            m = global[(y-1)*lenX+xoff-1+(originalN*originalN*threadID)];
            m = (m < prev) ? m : prev;
            m = (m == DBL_MAX) ? 0.0 : m;
            cur = distance(X[x-1+traceLowerResolutionIterator+(originalN*2*threadID)], Yy) + m;
            global[(y-1)*lenX+xoff-1+(originalN*originalN*threadID)] = cur;
            global[x-rightlowerX+y*lenX+(originalN*originalN*threadID)] = cur < prev ? cur : prev;
            prev = cur;
        }  

        for(int x = range3.start;x<range3.end;x++) {
            xoff = x-lowerX+1;
            m = global[(y-1)*lenX+xoff-1+(originalN*originalN*threadID)];
            m = (m < prev) ? m : prev;
            m = (m == DBL_MAX) ? 0.0 : m;
            cur = distance(X[x-1+traceLowerResolutionIterator+(originalN*2*threadID)], Yy) + m;
            global[(y-1)*lenX+xoff-1+(originalN*originalN*threadID)] = cur;
            prev = cur;
        }
        if ((upperX < rightupperX) && (y < lenY)) {
            global[upperX-rightlowerX+1+y*lenX+(originalN*originalN*threadID)] = cur;
        }
    }
    int warpPathLength = warppath(windowArray, originalN, lenX, global, C);
    return warpPathLength;
}

__device__ void visit(int x, int y, int row, int col, int* windowArray, int originalN) {
	int threadID =((blockIdx.x + gridDim.x * blockIdx.y) * (blockDim.x * blockDim.y)) + threadIdx.x + blockDim.x * threadIdx.y;
    windowArray[(y - 1)*2+(originalN*2*threadID)] = windowArray[(y - 1)*2+(originalN*2*threadID)] < (x == 0 ? 1 : x) ? windowArray[(y - 1)*2+(originalN*2*threadID)] : (x == 0 ? 1 : x);
    windowArray[(y - 1)*2+1+(originalN*2*threadID)] = windowArray[(y - 1)*2+1+(originalN*2*threadID)] > (x == row ? row : x + 1) ? windowArray[(y - 1)*2+1+(originalN*2*threadID)] : (x == row ? row : x + 1);

    if (y != col) {
        windowArray[y*2+(originalN*2*threadID)] = windowArray[y*2+(originalN*2*threadID)] < x ? windowArray[y*2+(originalN*2*threadID)] : x;
        windowArray[y*2+1+(originalN*2*threadID)] = windowArray[y*2+1+(originalN*2*threadID)] > x ? windowArray[y*2+1+(originalN*2*threadID)] : x;

        if (x != row) {
            windowArray[y*2+(originalN*2*threadID)] = windowArray[y*2+(originalN*2*threadID)] < (x + 1) ? windowArray[y*2+(originalN*2*threadID)] : (x + 1);
            windowArray[y*2+1+(originalN*2*threadID)] = windowArray[y*2+1+(originalN*2*threadID)] > (x + 1) ? windowArray[y*2+1+(originalN*2*threadID)] : (x + 1);
        }
    }
}

__device__ void expandWindow(int warpPathLength, int* C, int row, int cols, int newRow, int newCol, int radius, int originalN, int* windowArray){
	int threadID =((blockIdx.x + gridDim.x * blockIdx.y) * (blockDim.x * blockDim.y)) + threadIdx.x + blockDim.x * threadIdx.y;
    for (int i = 0; i < newRow; ++i) {
        for (int j = 0; j < 2; ++j) {
            if (j==0) {
                windowArray[i*2+j+(originalN*2*threadID)] = INT_MAX;
            }
            else {
                windowArray[i*2+j+(originalN*2*threadID)] = 0;
            }
        }
    }
    int x1 = 0;
    int y1 = 0;
    linear_to_2d_index(1, row, cols, &x1, &y1);
    int x2;
    int y2;
    int prev;
    int minX;
    int maxX;
    for(int i = warpPathLength; i >= 0;i--) {
        visit((x1-1)*2+1,(y1-1)*2+1,newRow, newCol ,windowArray, originalN);

        linear_to_2d_index(C[i+(originalN*2*threadID)], row, cols, &x2, &y2); 
        if ((x1 == x2) && (y1+1 == y2)) {

			//step right
			visit((x1-1)*2+1,(y1-1)*2+1+1,newRow, newCol,windowArray, originalN);
        }
        else if ((x1+1 == x2) && (y1 == y2)) {

			// step down
			visit((x1-1)*2+1+1,(y1-1)*2+1,newRow, newCol,windowArray, originalN);
        }
		else if ((x1+1 == x2) && (y1+1 == y2)) {

			// step diag
			visit((x1-1)*2+1+1,(y1-1)*2+1+1,newRow, newCol,windowArray, originalN);
        }
		x1 = x2;
		y1 = y2;

    }
	visit((x1-1)*2+1,(y1-1)*2+1,newRow, newCol, windowArray, originalN);
	visit((x1-1)*2+1+1,(y1-1)*2+1+1,newRow, newCol,windowArray, originalN);


	if ((y1-1)*2 +2 < newCol) {
        		visit((x1-1)*2+1,(y1-1)*2+1+1,newRow, newCol,windowArray, originalN);
    }
	prev = windowArray[originalN*2*threadID];
	if (radius >0) {
        for (int y = 1; y<newCol+1;y++) {
			if (y > 1) {
				minX = windowArray[(y-1)*2+(originalN*2*threadID)] - (windowArray[(y-1)*2+(originalN*2*threadID)] - prev) - radius;
				prev = windowArray[(y-1)*2+(originalN*2*threadID)];
				windowArray[(y-1)*2+(originalN*2*threadID)] = (1 > minX) ? 1 : minX;
            }

			if (y < newCol) {
				maxX = windowArray[(y-1)*2+1+(originalN*2*threadID)] + (windowArray[y*2+1+(originalN*2*threadID)] - windowArray[(y-1)*2+1+(originalN*2*threadID)]) + radius;
				windowArray[(y-1)*2+1+(originalN*2*threadID)] = (newRow < maxX) ? newRow : maxX;
            }
        }
    }
    return;
}

__device__ void resample(float* X, int lenPre, int lenAfter, int n, int traceLowerResolutionIterator, int originalN) {
    int threadID =((blockIdx.x + gridDim.x * blockIdx.y) * (blockDim.x * blockDim.y)) + threadIdx.x + blockDim.x * threadIdx.y;
    int nY = lenPre / n;
    float* Y = (float*)malloc(nY * sizeof(float)); 
    for (int x = 1; x <= nY; x++) {
        int o = (x - 1) * n;
        float sum = 0;
        for (int i = 0; i < n ; i++) {
            sum += X[o + i+traceLowerResolutionIterator-lenPre+(originalN*2*threadID)];
        }
        Y[x - 1] = sum / 2.0;
    }
    for (int i = 0;i<lenAfter;i++) {
        X[i+traceLowerResolutionIterator+(originalN*2*threadID)] = Y[i];
    }
    return;
}

extern "C"
__global__ void runFastDTW(float* A, float* B, int* traceLengths, int* windowArray ,int* C, int radius, int originalN, float* global) {
	int threadID =((blockIdx.x + gridDim.x * blockIdx.y) * (blockDim.x * blockDim.y)) + threadIdx.x + blockDim.x * threadIdx.y;

    int factor = 2;
    int minSize = radius + 4;
    int currentN = originalN;
    // Window array needs to be in global memory
    // Need to fill the window with the default window here
    defaultWindow(windowArray, originalN, originalN, originalN);
    
    
    int i = 0;
    int arraySize = 0;
    int z = 0;
    for(int h = originalN; h*2>minSize;h =h/factor) {
        traceLengths[z] = h;
        arraySize++;
        z++;
    }
    
    int totalElements = 0;
    for (int n=0; n < arraySize;n++) {
        totalElements += traceLengths[n];
    }
    i++;
    int traceLowerResolutionIterator = traceLengths[0];

    while ((currentN*2 > minSize)) {
        resample(&A[0], traceLengths[i-1], traceLengths[i],factor, traceLowerResolutionIterator, originalN);
        resample(&B[0], traceLengths[i-1], traceLengths[i],factor, traceLowerResolutionIterator, originalN);
        i++;
        traceLowerResolutionIterator += traceLengths[i-1];
        currentN = traceLengths[i];
    }
    traceLowerResolutionIterator -= traceLengths[i-1];
    if(minSize >= originalN) {
        int warpPathLength = dtw(A, B, windowArray, originalN, originalN, global, 0, originalN, C);
    }
    else {
        while(i >0) {
            int warpPathLength = dtw(A, B, windowArray, traceLengths[i-1],traceLengths[i-1], global, traceLowerResolutionIterator, originalN, C);
            // modify window so higher resolution can use it
            if(i >= 2) {
                expandWindow(warpPathLength, C, traceLengths[i-1], traceLengths[i-1], traceLengths[i-2], traceLengths[i-2], radius,originalN, windowArray);
            }    
            i = i-1;
            traceLowerResolutionIterator -= traceLengths[i-1];
        }
    }
}
'''

def createTRS(traceSet): # Used example from trsfile.readthedocs.io
	with trs_open(
		'output.trs',				 # File name of the trace set
		'w',							 # Mode: r, w, x, a (default to x)
		# Zero or more options can be passed (supported options depend on the storage engine)
		engine = 'TrsEngine',			# Optional: how the trace set is stored (defaults to TrsEngine)
		headers = {					  # Optional: headers (see Header class)
			Header.TRS_VERSION: 2,
			Header.SCALE_X: 1e-6,
			Header.SCALE_Y: 0.1,
			Header.DESCRIPTION: 'Testing trace creation',
			Header.TRACE_PARAMETER_DEFINITIONS: TraceParameterDefinitionMap(
				{'parameter': TraceParameterDefinition(ParameterType.BYTE, 16, 0)})
		},
		padding_mode = TracePadding.AUTO,# Optional: padding mode (defaults to TracePadding.AUTO)
		live_update = True			   # Optional: updates the TRS file for live preview (small performance hit)
										 #   0 (False): Disabled (default)
										 #   1 (True) : TRS file updated after every trace
										 #   N		: TRS file is updated after N traces
	) as traces:
            for trace in (traceSet):
                traces.extend([
                Trace(
                    SampleCoding.FLOAT,
                    trace,
                    TraceParameterMap({'parameter': ByteArrayParameter(os.urandom(16))})
                )]
            )
            print('Amount of traces in the new trace set: {0:d}'.format(len(traces)))

def runFastDTWKernel(referenceTrace, traceSet1, radius, amountOfThreads, traceLength):
    # Variables
    N = traceLength

    traceSizeWithLowerResolutions = N * 2 * amountOfThreads# The amount of samples for the trace + lower resolutions
    traceSizeWithLowerResolutionsMemory = traceSizeWithLowerResolutions * np.dtype(np.float32).itemsize # The memory in bytes for ^

    # Array to store the lengths of all the arrays and the lower resolution arrays, not sure on the size needed. log(N) size needed per thread.
    traceLengths =  N * amountOfThreads
    traceLengthsMemory = traceLengths * np.dtype(np.int32).itemsize

    # Result array
    sizePath = N * 2 * amountOfThreads 
    sizePathMemory = sizePath * np.dtype(np.int32).itemsize

    # 2D cost matrix for dtw
    matrixSize = N*N
    sizeGlobal = matrixSize*amountOfThreads
    sizeGlobalMemory = sizeGlobal * np.dtype(np.float32).itemsize

    # Window
    windowGlobal = N*2*amountOfThreads
    windowGlobalMemory = windowGlobal * np.dtype(np.int32).itemsize

    # Extra variables
    shrinkFactor = 2

    # Finding NVIDIA GPU
    devID = 0 # GPU ID
    checkCudaErrors(cuda.cuInit(devID))
    cuDevice = findCudaDeviceDRV()
    cuContext = checkCudaErrors(cuda.cuCtxCreate(0, cuDevice))
    uvaSupported = checkCudaErrors(cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, cuDevice))
    if not uvaSupported:
        print("Accessing pageable memory directly requires UVA")
        return

    # Getting the C kernel code
    kernelHelper = common.KernelHelper(fastdtwDrv, int(cuDevice))
    _VecAdd_kernel = kernelHelper.getFunction(b'runFastDTW')


    # Creating arrays and allocating memory to copy to the device (GPU).

    # Creating two arrays with traces followed by the same amount of 0's to store lower resolutions.
    iterateFrom = 0
    iterateTo = N
    referenceTraceSet = np.empty(shape=(0,), dtype=np.float32)
    trace1NP1 = np.empty(shape=(0,), dtype=np.float32)
    for x in range(amountOfThreads):
        referenceTrace = np.array(referenceTrace, dtype=np.float32)
        trace1NP = np.array(traceSet1[iterateFrom:iterateTo], dtype=np.float32)
        referenceTraceSet = np.concatenate((referenceTraceSet, referenceTrace, np.full(N, np.float32(0.0))))
        trace1NP1 = np.concatenate((trace1NP1, trace1NP, np.full(N, np.float32(0.0))))
        iterateFrom += N
        iterateTo += N

 	# Allocate array h_A and h_B, that will contain the traces and space for lower resolution traces, for every thread.
    h_A = referenceTraceSet
    h_B = trace1NP1


    # Allocate array h_traceLengths as space to put the lengths of lower resolutions, for every thread.
    h_traceLengths = np.full(traceLengths,0)

    # Allocate array h_window as space for window management, for every thread.
    h_window = np.full(windowGlobal, 0)

    # Allocate array h_result as space to store the warp paths in, for every thread.
    h_result = np.random.rand(sizePath).astype(dtype=np.int32)

    # Allocate array h_global as space for the cost matrix, for every thread.
    h_global = np.full(sizeGlobal, np.finfo(np.float32).max) 

	# Allocate arrays in device memory
    d_A = checkCudaErrors(cuda.cuMemAlloc(traceSizeWithLowerResolutionsMemory))
    d_B = checkCudaErrors(cuda.cuMemAlloc(traceSizeWithLowerResolutionsMemory))
    d_traceLengths = checkCudaErrors(cuda.cuMemAlloc(traceLengthsMemory))
    d_window = checkCudaErrors(cuda.cuMemAlloc(windowGlobalMemory))
    d_result = checkCudaErrors(cuda.cuMemAlloc(sizePathMemory))
    d_global = checkCudaErrors(cuda.cuMemAlloc(sizeGlobalMemory))


	# Copy vectors from host memory to device memory
    checkCudaErrors(cuda.cuMemcpyHtoD(d_A, h_A, traceSizeWithLowerResolutionsMemory))
    checkCudaErrors(cuda.cuMemcpyHtoD(d_B, h_B, traceSizeWithLowerResolutionsMemory))
    checkCudaErrors(cuda.cuMemcpyHtoD(d_traceLengths, h_traceLengths, traceLengthsMemory))
    checkCudaErrors(cuda.cuMemcpyHtoD(d_window, h_window, windowGlobalMemory))
    checkCudaErrors(cuda.cuMemcpyHtoD(d_global, h_global, sizeGlobalMemory))

    if True:
		# Grid/Block configuration
        threadsPerBlock = 4
        blocksPerGrid   = amountOfThreads / threadsPerBlock

        kernelArgs = ((d_B, d_A, d_traceLengths,d_window,d_result, radius, N, d_global),
					  (None, None, None, None, None, ctypes.c_int, ctypes.c_int, None))
		
		# Launch the CUDA kernel
        checkCudaErrors(cuda.cuLaunchKernel(_VecAdd_kernel,
											amountOfThreads, 1, 1,
											1, 1, 1,
											0, 0,
											kernelArgs, 0))
		
    else:
        pass

	# Copy result from device memory to host memory
	# h_C contains the result in host memory
    cudart.cudaDeviceSynchronize()
    checkCudaErrors(cuda.cuMemcpyDtoH(h_result, d_result, sizePathMemory))

	# Free device memory
    checkCudaErrors(cuda.cuMemFree(d_A))
    checkCudaErrors(cuda.cuMemFree(d_B))
    checkCudaErrors(cuda.cuMemFree(d_result))
    checkCudaErrors(cuda.cuCtxDestroy(cuContext))
    
    return h_result

def linearToCar(path, dims):
    cartindxes = np.array(list(np.ndindex(dims))) + 1

    newPath = [0] * len(path)
    for x, n in enumerate(path):
        newPath[x] = cartindxes[n-1]
    return newPath

def linear_to_2d_index(linear_index, num_rows, num_cols):
    x = ((linear_index - 1) % num_cols) + 1
    y = ((linear_index - 1) // num_cols) + 1
    return x, y


def linearToCarFast(path, length):
    newPath = [0] * len(path)
    for x, n in enumerate(path):
        newPath[x] = linear_to_2d_index(path[x], length, length)
    return newPath



def createNewTrace(trace0, trace1, path):
    averages = {}
    for point in path:
        y, x = point
        if x not in averages:
            averages[x] = []
        if y - 1 < len(trace1):
            averages[x].append(trace1[y - 1])

    result = {x: sum(vals) / len(vals) if len(vals) > 0 else 0 for x, vals in averages.items()}
    result = [float(value) for value in result.values()]
    return result

def distance(x, y):
    return (x - y) * (x - y)

def totalTraceDistance(trace0, trace1):
    sum = 0.0
    for x in range(len(trace0)):
        sum += distance(trace0[x], trace1[x])
    return sum


def main():
    radius = 100 # Radius to run algorithm with
    amountOfThreads = amountOfTraces
    traceLength = len(referenceTrace)
    print("Amount of samples in the reference trace:", traceLength)
    # Calling the function that runs cuda
    result = runFastDTWKernel(referenceTrace, traceSet, radius, amountOfThreads, traceLength)

    np.set_printoptions(threshold=np.inf)

    # To get the individual warp paths of every thread.
    result = np.array_split(result, amountOfThreads)

    # To remove the elements that do not contain the warp path, namely the elements 0
    traceResults = [np.trim_zeros(subarr, 'b') for subarr in result]

    # Reverse the warp path to start from the beginning of the traces instead of the end.
    reversedTraceResults = [list(reversed(subarray)) for subarray in traceResults]

    # To convert the warp path in to coordinates instead of n'th element of the cost matrix.
    traceResultsCar = [list(linearToCarFast(subarray, traceLength)) for subarray in reversedTraceResults]

    # To convert the warp paths in to new aligned traces.
    newTraceSet = []
    for x, trace in enumerate(traceResultsCar):
        newTraceSet.append(list(createNewTrace(referenceTrace, traceSet[x*traceLength:x*traceLength+traceLength],trace)))

    # Creating an output trs file with the newly obtained traces.
    createTRS(newTraceSet)

    b = time.time()
    c = b - a
    print("runtime (s):", c)

if __name__ == "__main__":
	main()