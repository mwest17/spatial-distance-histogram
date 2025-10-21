/* ==================================================================
	Programmer: Yicheng Tu (ytu@cse.usf.edu)
	The basic SDH algorithm implementation for 3D data
	To compile: nvcc SDH.c -o SDH in the GAIVI machines
   ==================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>


#define BOX_SIZE	23000 /* size of the data box on one dimension            */
#define ADDITION_CYCLES 73

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

// typedef double3 gpu_atom;

typedef struct atomdesc_gpu{ 
	double* x;
	double* y;
	double* z;
} gpu_atom;

typedef struct hist_entry{
	unsigned long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;

typedef struct gpu_hist_entry{
	unsigned long d_cnt;
} gpu_bucket;


bucket * histogram;		/* list of all buckets in the histogram     */
bucket* gpu_histogram;  /* list of all buckets in the GPU histogram */
long long	PDH_acnt;	/* total number of data points              */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w (the width of each bucket)    */
atom * atom_list;		/* list of all data points                  */
gpu_atom gpuAtoms;      /* list of data points in GPU's format      */
double p[32]; 			/* Probability of no collisions             */


/* These are for an old way of tracking time */
struct timezone Idunno;
struct timeval startTime, endTime;


/*
	distance of two points in the atom_list
*/
double p2p_distance(int ind1, int ind2) {

	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;

	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


/*
	brute-force SDH solution in a single CPU thread
*/
int PDH_baseline() {
	int i, j, h_pos;
	double dist;

	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distance(i,j);
			h_pos = (int) (dist / PDH_res);
			histogram[h_pos].d_cnt++;
		}
	}
	return 0;
}


//##############################################################################
// GPU Code
//##############################################################################

__device__ inline double reciprocal_sqrt(double x) {
    double y = rsqrt(x);
    return x * y;
}

__device__ inline double euclidDist(double3 p1, double p2x, double p2y, double p2z)
{
	// Component distances between p1 and p2
	double dx = p1.x - p2x;
	double dy = p1.y - p2y;
	double dz = p1.z - p2z;

	// Straight line distance between points
	return reciprocal_sqrt(dx*dx + dy*dy + dz*dz);
}

/*
	GPU kernel function to compute the PDH for a given set of 3d points
*/
// **TODO** Need to figure out how to do this:
	// Moreover, we vectorize each dimension array by loading multiple floating point coordinate values in one data transmission unit
__global__ void PDH_kernel(gpu_atom dev_atom_list, // Array containing all datapoints
					  bucket* dev_histogram, // Array of bucket counts
					  const int PDH_acnt, // Number of datapoints
					  const int PDH_res, // Bucket size
					  const int num_buckets,
					  const int numHistograms)
{
	extern __shared__ unsigned char sharedMemory[];

	gpu_atom tile;
	tile.x = (double*)sharedMemory;
	tile.y = (double*)sharedMemory + blockDim.x;
	tile.z = (double*)sharedMemory + 2 * blockDim.x;

	int warpOffset = threadIdx.x & 0x1f;
	int histOffset = num_buckets*(warpOffset % numHistograms);
	gpu_bucket* localHist = (gpu_bucket*) ((double*)sharedMemory + 3 * blockDim.x);

	// Initialize local histogram to 0
	for (unsigned i = threadIdx.x; i < num_buckets * numHistograms; i += blockDim.x)
	{
		localHist[i].d_cnt = 0;
	}

	// Check if our current thread index is out of range of the array
	unsigned long long int index = (blockDim.x * blockIdx.x) + threadIdx.x;
	
	// Load this thread's left point into a register
	double3 localPoint;
	if (index < PDH_acnt) {
		localPoint.x = dev_atom_list.x[index];
		localPoint.y = dev_atom_list.y[index];
		localPoint.z = dev_atom_list.z[index];
	}
	else {
		localPoint.x = 0; localPoint.y = 0; localPoint.z = 0;
	}

	for (unsigned long int tileInd = blockIdx.x + 1; tileInd < gridDim.x; tileInd++)
	{
		// Load next tile into shared memory
		unsigned long int tileIndex = (blockDim.x * tileInd) + threadIdx.x;
		if (tileIndex < PDH_acnt)
		{
			tile.x[threadIdx.x] = dev_atom_list.x[tileIndex];
			tile.y[threadIdx.x] = dev_atom_list.y[tileIndex];
			tile.z[threadIdx.x] = dev_atom_list.z[tileIndex];
		}	
		__syncthreads();
		

		// Find distance from thread's point to all points in tile
		for (int i = 0; i < blockDim.x; i++)
		{	
			unsigned long long int ind = (blockDim.x * tileInd) + i;
			if (ind < PDH_acnt) {
				// Straight line distance between points
				double dist = euclidDist(localPoint, tile.x[i], tile.y[i], tile.z[i]);
			
				// Determine which bucket it should go into
				int bucket = (int) (dist / PDH_res);
				
				atomicAdd((unsigned long long *) &(localHist[histOffset + bucket].d_cnt), (unsigned long long) 1);
			}
		}
		__syncthreads();
		
	}

	// Every thread store its assigned point into tile
	tile.x[threadIdx.x] = localPoint.x;
	tile.y[threadIdx.x] = localPoint.y;
	tile.z[threadIdx.x] = localPoint.z;
	__syncthreads();

	// Find intra point distances
	// **TODO** Balance the intra point distance calculation
	// Balancing is causing some histogram blocks to count more. Almost all is clustered in specific ones. 
	// We myst be iterating too many times.
	// for (int i = 1; i <= blockDim.x / 2; i++) 
	// {
	// 	int tileIndex = (threadIdx.x + i) % blockDim.x;
	// 	unsigned long long int ind = (blockDim.x * blockIdx.x) + tileIndex;
	// 	if (ind < PDH_acnt && (i <= blockDim.x / 2 - 1 || threadIdx.x < (blockDim.x / 2)))
	// 	{
	// 		double dist = euclidDist(localPoint, tile.x[i], tile.y[i], tile.z[i]);
	// 		int bucket = (int) (dist / PDH_res);
	// 		atomicAdd((unsigned long long *) &(localHist[histOffset + bucket].d_cnt), (unsigned long long) 1);
	// 	}
	// }

	for (int i = threadIdx.x + 1; i < blockDim.x; i++) 
	{
		unsigned long long int ind = (blockDim.x * blockIdx.x) + i;

		if (ind < PDH_acnt)
		{
			double dist = euclidDist(localPoint, tile.x[i], tile.y[i], tile.z[i]);
			int bucket = (int) (dist / PDH_res);

			atomicAdd((unsigned long long *) &(localHist[histOffset + bucket].d_cnt), (unsigned long long) 1);
		}
	}

	__syncthreads();


	// Merging private histogram copies into a single copy
	for (unsigned int curBucket = 0; curBucket < num_buckets; curBucket++)
	{
		for (unsigned int stride = numHistograms/2; stride > 0; stride /= 2) 
		{
			if (threadIdx.x < stride)
			{
				localHist[curBucket + num_buckets*threadIdx.x].d_cnt += localHist[curBucket + num_buckets*threadIdx.x + stride*num_buckets].d_cnt; 
			}
			__syncthreads();
		}
	}

	// Copy local output to global memory
	for (int i = threadIdx.x; i < num_buckets; i += blockDim.x)
	{
		// **TODO** Use a faster tree based reduction algorithm
		// atomicAdd(&(dev_histogram[i].d_cnt), (unsigned long long) localHist[i].d_cnt);
		dev_histogram[blockIdx.x * num_buckets + i].d_cnt = localHist[i].d_cnt;
	}

	// Parallel reduction

	

	for (unsigned int curBucket = blockIdx.x; curBucket < num_buckets; curBucket += gridDim.x)
	{
		for (unsigned int stride = numHistograms/2; stride > 0; stride /= 2) // Not enough threads for every bucket
		{
			if (threadIdx.x < stride)
			{
				dev_histogram[curBucket + num_buckets*threadIdx.x].d_cnt += dev_histogram[curBucket + num_buckets*threadIdx.x + stride*num_buckets].d_cnt; 
			}
			__syncthreads();
		}
	}
	// unsigned int i = blockIdx.x + num_buckets * threadIdx.x;
	// unsigned int ri = blockIdx.x + 2 * num_buckets * threadIdx.x;  
	// localHist[threadIdx.x].d_cnt = dev_histogram[i].d_cnt + dev_histogram[ri].d_cnt;
	// // Local hist is a gpu_bucket, so will not work.

	// __syncthreads();
	// for (unsigned int stride = gridDim.x / 2; stride > 0; stride /= 2)
	// {
	// 	if (threadIdx.x < stride)
	// 	{
	// 		localHist[threadIdx.x].d_cnt += localHist[threadIdx.x + stride].d_cnt;
	// 	} 
	// 	__syncthreads();
	// }
	// if (threadIdx.x == 0) dev_histogram[blockIdx.x].d_cnt = localHist[0].d_cnt;

}

__global__ void reduction(bucket* dev_histogram, // Array of bucket counts
					  const int num_buckets,
					  const int numHistograms)
{
	// Every Block reduces 1 index of the histogram
	// Offset is then blockIdx.x
	// unsigned int segment = 2*blockDim.x*blockIdx.x;
	// printf("Test\n");
	unsigned int i = blockIdx.x + num_buckets * threadIdx.x;
	unsigned int ri = blockIdx.x + 2 * num_buckets * threadIdx.x;  
	// Just need to figure out the i value, 
	// then rest of reduction should work fine

	__shared__ bucket input_s[64];
	input_s[threadIdx.x].d_cnt = dev_histogram[i].d_cnt + dev_histogram[ri].d_cnt;
	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		if (threadIdx.x < stride)
		{
			input_s[threadIdx.x].d_cnt += input_s[threadIdx.x + stride].d_cnt;
		}
		__syncthreads();
	}	

	//Final sum for that bucket will be in index 0
	if (threadIdx.x == 0) atomicAdd(&dev_histogram[blockIdx.x % num_buckets].d_cnt, (unsigned long long) input_s[0].d_cnt);
}


double findLatency(const int k, const int cl = ADDITION_CYCLES)
{
	if (k == 1) return cl;
	return p[k]*cl + (1.5-p[k])*findLatency(k-1, cl + ADDITION_CYCLES); // **TODO** I don't like this
	// Want to find a way to make collisions more impactful without subtracting from 1.5
}

int findRounds(const unsigned int blockSize, const unsigned long long int numBlocks, const size_t sizeHistogram, int k)
{
	int numHistograms = 32 / k;

	int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

	int blocksPerSM;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, PDH_kernel, blockSize, numHistograms * sizeHistogram + 3 * sizeof(double) * blockSize);
	
	//number of threads that can be run in each round in a single multiprocessor 
	unsigned long int occupancy = blocksPerSM * blockSize;
	
	int numThreads = blockSize * numBlocks;
	int numMultiprocessors = prop.multiProcessorCount;

	double denominator = numMultiprocessors * occupancy;
	return (numThreads + denominator - 1 ) / denominator;
}

/*
	Finds the optimal number of histogram copies based on latency and and occupancy
*/
int findNumHistograms(const unsigned int blockSize, const unsigned long long int numBlocks, const size_t sizeHistogram) 
{
	int bestk;
	double minLR = INFINITY;

	for (int k = 1; k <= 32; k++) { // Initialize probabilities
		p[k] = exp(-(k*(k-1))/(double)(2*num_buckets));
		// printf("p[%d] = %lf\n", k, p[k]);
	}

	for (int k = 1; k <= 32; k *= 2)
	{
		// L Value
		double L = findLatency(k);

		// R Value
		int R = findRounds(blockSize, numBlocks, sizeHistogram, k); 

		// Find L x R (Total time)
		double LR = L * R;
		
		// printf("NumHist: %d, L: %lf R: %d, LxR: %lf\n", 32 / k, L, R, LR);

		// Find if LxR is smaller than current min
		if (LR > 0 && LR < minLR)
		{
			minLR = LR;
			bestk = k;
		}
	}

	return 32 / bestk;
}


/*
	Wrapper for the PDH gpu kernel function
	Returns the time taken to run CUDA kernel
*/
float PDH_gpu(const unsigned int blockSize = 64)
{
	const size_t sizeAtomList = sizeof(double)*PDH_acnt;
	const size_t sizeHistogram = sizeof(gpu_bucket)*num_buckets;

	// Allocating Memory
	gpu_atom dev_atom_list;
	cudaMalloc((void**) &(dev_atom_list.x), sizeAtomList);
	cudaMalloc((void**) &(dev_atom_list.y), sizeAtomList);
	cudaMalloc((void**) &(dev_atom_list.z), sizeAtomList);

	// Copying input values to gpu atom list
	cudaMemcpy(dev_atom_list.x, gpuAtoms.x, sizeAtomList, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_atom_list.y, gpuAtoms.y, sizeAtomList, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_atom_list.z, gpuAtoms.z, sizeAtomList, cudaMemcpyHostToDevice);

	// Need 1 thread per point
	const unsigned long long int numBlocks = (PDH_acnt + blockSize - 1) / blockSize;

	// printf("Size of the histogram: %ld\n", sizeHistogram);
	int numHistograms = findNumHistograms(blockSize, numBlocks, sizeHistogram);
	size_t amountSharedMemory = sizeHistogram * numHistograms + 3 * sizeof(double) * blockSize; 
	// printf("Num hist: %d\n", numHistograms);

	bucket* dev_histogram;
	cudaMalloc((void**) &dev_histogram, sizeHistogram * numBlocks);
	cudaMemset(dev_histogram, 0, sizeHistogram * numBlocks);

	// Start timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	PDH_kernel<<<numBlocks, blockSize, amountSharedMemory>>>
		(dev_atom_list, dev_histogram, PDH_acnt, PDH_res, num_buckets, numHistograms);

	// Record end time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// Calculate total time spent computing
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copy output histogram from global to cpu mem
	cudaMemcpy(gpu_histogram, dev_histogram, sizeHistogram, cudaMemcpyDeviceToHost);
	cudaFree(dev_atom_list.x);
	cudaFree(dev_atom_list.y);
	cudaFree(dev_atom_list.z);
	cudaFree(dev_histogram);

	return elapsedTime;
}

//##############################################################################


/*
	set a checkpoint and show the (natural) running time in seconds
*/
double report_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("\nRunning time for CPU version (in seconds): %ld.%06ld", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}

double report_gpu_running_time(float elapsedTimeMS) {
	// Convert miliseconds to seconds
	double elapsedTimeS = elapsedTimeMS / 1000.0;
	printf("\nRunning time for GPU version (in seconds): %lf", elapsedTimeS);
	return elapsedTimeS;
}


/*
	print the counts in all buckets of the histogram
*/
void output_histogram(){
	int i;
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram[i].d_cnt);
		total_cnt += histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}

void gpu_output_histogram(){
	int i;
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", gpu_histogram[i].d_cnt);
		total_cnt += gpu_histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}


/*
	Compute and display the difference between the CPU and GPU histograms
*/
void compare_histograms(bucket *cpu_hist, bucket *gpu_hist) {
    bool different = false;
	printf("\nDifference between CPU and GPU histograms:");
    for (int i = 0; i < num_buckets; i++) {
        long long diff = cpu_hist[i].d_cnt - gpu_hist[i].d_cnt;
		if (diff != 0) different = true;
        if (i % 5 == 0)
            printf("\n%02d: ", i);
        printf("%15lld ", diff);
        if (i != num_buckets - 1)
            printf("| ");
    }
    printf("\n");
	(different)? printf("Different\n") : printf("Not different\n");
}


int main(int argc, char **argv)
{
	int i;

	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);
	int blockSize = atoi(argv[3]);
// printf("args are %d and %f\n", PDH_acnt, PDH_res);

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	gpu_histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);

	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);

	gpuAtoms.x = (double*)malloc(sizeof(double)*PDH_acnt);
	gpuAtoms.y = (double*)malloc(sizeof(double)*PDH_acnt);
	gpuAtoms.z = (double*)malloc(sizeof(double)*PDH_acnt);

	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		gpuAtoms.x[i] = atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		gpuAtoms.y[i] = atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		gpuAtoms.z[i] = atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}
	/* start counting time */
	gettimeofday(&startTime, &Idunno);

	/* call CPU single thread version to compute the histogram */
	PDH_baseline();

	/* check the total running time */
	report_running_time();

	/* print out the histogram */
	output_histogram();

	/* Computing histograms on GPU */
	float elapsedTime = PDH_gpu(blockSize);

	report_gpu_running_time(elapsedTime);

	gpu_output_histogram();

	/* Compare histograms between cpu and gpu */
	compare_histograms(histogram, gpu_histogram);

	return 0;
}

