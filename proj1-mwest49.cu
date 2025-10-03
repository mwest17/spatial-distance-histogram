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

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	//float min;
	//float max;
	unsigned long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;


bucket * histogram;		/* list of all buckets in the histogram   */
bucket* gpu_histogram;  /* GPU output bucket list*/
long long	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom * atom_list;		/* list of all data points                */

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

/*
	GPU kernel function to compute the PDH for a given set of 3d points
*/
__global__ void PDH_kernel(atom* dev_atom_list, // Array containing all datapoints
						   bucket* dev_histogram, // Array of bucket counts
						   int PDH_acnt, // Number of datapoints
						   int PDH_res) // Bucket size 
{
	// Check if our current thread index is out of range of the array	
	int index = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (index < PDH_acnt) 
	{
		// Compute distance between points in the range [index + 1....n]
		for (int i = index + 1; i < PDH_acnt; i++)
		{
			// Compute component distances from point at index to point at i
			double dx = dev_atom_list[index].x_pos - dev_atom_list[i].x_pos;
			double dy = dev_atom_list[index].y_pos - dev_atom_list[i].y_pos;
			double dz = dev_atom_list[index].z_pos - dev_atom_list[i].z_pos;

			// Straight line distance between points
			double dist = sqrt(dx*dx + dy*dy + dz*dz);
			
			// Determine which bucket it should go into
			int bucket = (int) (dist / PDH_res);
			
			atomicAdd(&(dev_histogram[bucket].d_cnt), (unsigned long long) 1);
		}
	}
}


/*
	Wrapper for the PDH gpu kernel function
*/
int PDH_gpu() 
{
	const size_t sizeAtomList = sizeof(atom)*PDH_acnt;
	const size_t sizeHistogram = sizeof(bucket)*num_buckets;
	
	// Allocating Memory
	atom* dev_atom_list;
	cudaMalloc((void**) &dev_atom_list, sizeAtomList);
	bucket* dev_histogram;
	cudaMalloc((void**) &dev_histogram, sizeHistogram);

	// Copying input values to gpu atom list
	cudaMemcpy(dev_atom_list, atom_list, sizeAtomList, cudaMemcpyHostToDevice);
	cudaMemset(dev_histogram, 0, sizeHistogram);

	// Need 1 thread per point
	int threadsPerBlock = 256;
	int blocksPerGrid = (PDH_acnt + threadsPerBlock - 1) / threadsPerBlock;

	// Call kernel function (Passing in the array of data)
	PDH_kernel<<<blocksPerGrid, threadsPerBlock>>>(dev_atom_list, dev_histogram, PDH_acnt, PDH_res);

	// Copy output histogram from global to cpu mem
	cudaMemcpy(gpu_histogram, dev_histogram, sizeHistogram, cudaMemcpyDeviceToHost);

	cudaFree(dev_atom_list);
	cudaFree(dev_histogram);

	return 0;
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
	printf("Running time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}

double report_gpu_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for GPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
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
    printf("\nDifference between CPU and GPU histograms:\n");
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
// printf("args are %d and %f\n", PDH_acnt, PDH_res);

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	gpu_histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);

	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);
	
	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
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
	gettimeofday(&startTime, &Idunno);

	PDH_gpu();

	report_gpu_running_time();

	gpu_output_histogram();

	/* Compare histograms between cpu and gpu */
	compare_histograms(histogram, gpu_histogram);

	return 0;
}

