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

#include <cuda_runtime.h>


#define BOX_SIZE	23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef double3 gpu_atom;

typedef struct hist_entry{
	unsigned long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;


bucket * histogram;		/* list of all buckets in the histogram     */
bucket* gpu_histogram;  /* list of all buckets in the GPU histogram */
long long	PDH_acnt;	/* total number of data points              */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w (the width of each bucket)    */
atom * atom_list;		/* list of all data points                  */
gpu_atom* gpuAtoms;     /* list of data points in GPU's format      */


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

__device__ inline double euclidDist(double3 p1, double3 p2)
{
	// Component distances between p1 and p2
	double dx = p1.x - p2.x;
	double dy = p1.y - p2.y;
	double dz = p1.z - p2.z;

	// Straight line distance between points
	return sqrt(dx*dx + dy*dy + dz*dz);
}

/*
	GPU kernel function to compute the PDH for a given set of 3d points
*/
// Potential optimizations:
// Store copy of output historgram in each of the warps' shared memory

// Threads are not in sync at all. Need to figure out way to make all perform roughly same number of work
// Number of computations range from n to 1

// Going to want to divide distances into sections and figure out how to iterate through

// Threads divide up input into sections and move down each section

__global__ void PDH_kernel(gpu_atom* dev_atom_list, // Array containing all datapoints
						   bucket* dev_histogram, // Array of bucket counts
						   int PDH_acnt, // Number of datapoints
						   int PDH_res) // Bucket size
{
	__shared__ double3 tile[blockDim.x];

	// **TODO** Output privitization in shared memory

	// Check if our current thread index is out of range of the array
	int index = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (index < PDH_acnt)
	{
		// Load this thread's left point into a register
		double3 localPoint = dev_atom_list[index];

		for (int i = blockIdx.x + 1; i < gridDim.x; i++)
		{
			// Load next tile into shared memory
			tile[threadIdx.x] = dev_atom_list[index];
			__syncthreads();

			for (int j = 0; j < blockDim.x; j++)
			{
				// Straight line distance between points
				double dist = euclidDist(localPoint, tile[j]);
				
				// Determine which bucket it should go into
				int bucket = (int) (dist / PDH_res);

				atomicAdd(&(dev_histogram[bucket].d_cnt), (unsigned long long) 1);
			}
		}

		// Find intra point distances
		tile[threadIdx.x] = localPoint;
		__syncthreads();

		// **TODO** Balance the intra point distance calculation
		for (i = threadIdx.x + 1; i < blockDim.x; i++) 
		{
			double dist = euclidDist(localPoint, tile[i]);
			int bucket = (int) (dist / PDH_res);

			atomicAdd(&(dev_histogram[bucket].d_cnt), (unsigned long long) 1);
		}
		__syncthreads();
	}
}


/*
	Wrapper for the PDH gpu kernel function
	Returns the time taken to run CUDA kernel
*/
float PDH_gpu(const int blockSize = 256)
{
	const size_t sizeAtomList = sizeof(gpu_atom)*PDH_acnt;
	const size_t sizeHistogram = sizeof(bucket)*num_buckets;

	// Allocating Memory
	gpu_atom* dev_atom_list;
	cudaMalloc((void**) &(dev_atom_list), sizeAtomList);

	// Copying input values to gpu atom list
	cudaMemcpy(dev_atom_list, gpuAtoms, sizeAtomList, cudaMemcpyHostToDevice);

	bucket* dev_histogram;
	cudaMalloc((void**) &dev_histogram, sizeHistogram);
	cudaMemset(dev_histogram, 0, sizeHistogram);

	// Need 1 thread per point
	const int numBlocks = (PDH_acnt + blockSize - 1) / blockSize;

	// Start timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Call kernel function (Passing in the array of data)
	PDH_kernel<<<numBlocks, blockSize>>>(dev_atom_list, dev_histogram, PDH_acnt, PDH_res);

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

	cudaFree(dev_atom_list);
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
    printf("\nDifference between CPU and GPU histograms:");
    for (int i = 0; i < num_buckets; i++) {
        long long diff = cpu_hist[i].d_cnt - gpu_hist[i].d_cnt;
        if (i % 5 == 0)
            printf("\n%02d: ", i);
        printf("%15lld ", diff);
        if (i != num_buckets - 1)
            printf("| ");
    }
    printf("\n");
}


int main(int argc, char **argv)
{
	int i;

	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);
	// int blockSize = atoi(argv[3]);
// printf("args are %d and %f\n", PDH_acnt, PDH_res);

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	gpu_histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);

	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);

	gpuAtoms = (gpu_atom*)malloc(sizeof(gpu_atom)*PDH_acnt);

	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;

		gpuAtoms[i].x = atom_list[i].x_pos;
		gpuAtoms[i].y = atom_list[i].y_pos;
		gpuAtoms[i].z = atom_list[i].z_pos;
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
	float elapsedTime = PDH_gpu();

	report_gpu_running_time(elapsedTime);

	gpu_output_histogram();

	/* Compare histograms between cpu and gpu */
	compare_histograms(histogram, gpu_histogram);

	return 0;
}

