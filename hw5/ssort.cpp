// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>
#include <numeric>
#include <iostream>

template <class T>
void printv(T* arr, int N) {
    for (int i = 0; i < N; i++) {
        std::cout << arr[i] << ' ';
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    // Number of random numbers per processor (this should be increased
    // for actual tests or could be passed in through the command line
    int N = 100;

    if (argc != 2) {
        printf("require a N\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
    sscanf(argv[1], "%d", &N);

    int *vec = (int *) malloc(N * sizeof(int));
    int *splitter = (int *) malloc((p - 1) * sizeof(int));

    // for better optimization, only root would allocate this ???? but how does the MPI_GATHER reference this?
    int *global_splitter = (int *) malloc(p * (p - 1) * sizeof(int));

    // pointer to end address of every segments
    // p_i owns ends[i-1] to ends[i]

    // seed random number generator differently on every core
    srand((unsigned int) (rank + 3939199));

    // fill vector with random integers
    for (int i = 0; i < N; ++i) {
        vec[i] = rand() % (p*N);
    }

    // sort locally
    std::sort(vec, vec + N);

    double tt = MPI_Wtime();
    // sample p-1 entries from vector as the local splitters, i.e.,
    // every N/P-th entry of the sorted vector
    for (int i = 0; i < p-1; i++) {
        splitter[i] = vec[(N+p-1)/p * (i+1) -1];
    }
//    std::cout << rank << " vec ";
//    printv(vec, N);
//    std::cout<< rank << " splitter ";
//    printv(splitter, p-1);

    // every process communicates the selected entries to the root
    // process; use for instance an MPI_Gather
    MPI_Gather(splitter, p - 1, MPI_INT, global_splitter, (p - 1), MPI_INT, 0, MPI_COMM_WORLD);

    // root process does a sort and picks (p-1) splitters (from the
    // p(p-1) received elements)

    if (rank == 0) {
        std::sort(global_splitter, global_splitter+p*(p-1));

        // do sampling, select p-1 as pivots
        for (int i = 0; i < p-1; i++) {
            splitter[i] = global_splitter[(p-1) * (i+1) - 1];
        }
//        std::cout << "global splitter ";
//        printv(global_splitter, p*(p-1));
//        std::cout << " sample ";
//        printv(splitter, (p-1));
    }

    // root process broadcasts splitters to all other processes
    MPI_Bcast(splitter, p-1, MPI_INT, 0, MPI_COMM_WORLD);

    // every process uses the obtained splitters to decide which
    // integers need to be sent to which other process (local bins).
    // Note that the vector is already locally sorted and so are the
    // splitters; therefore, we can use std::lower_bound function to
    // determine the bins efficiently.
    //
    // Hint: the MPI_Alltoallv exchange in the next step requires
    // send-counts and send-displacements to each process. Determining the
    // bins for an already sorted array just means to determine these
    // counts and displacements. For a splitter s[i], the corresponding
    // send-displacement for the message to process (i+1) is then given by,
    // sdispls[i+1] = std::lower_bound(vec, vec+N, s[i]) - vec;

    // send and receive: first use an MPI_Alltoall to share with every
    // process how many integers it should expect, and then use
    // MPI_Alltoallv to exchange the data

    // first share your length to world
    int* start = vec;
    int send_offset[p];
    int send_counts[p];
    int* end;
    for (int i = 0; i < p; ++i) {
        // p_0 to p_(p-1)
        send_offset[i] = start-vec; // offset from vector begin
        end = i == p-1? vec+N: std::lower_bound(vec, vec+N, splitter[i]);
        send_counts[i] = end - start; // 32 bit is ok, for 64 bit os might cause problem
        start = end;
    }

//    std::cout<< rank << " send offsets ";
//    printv(send_offset, p);
//    std::cout<< rank << " send counts ";
//    printv(send_counts, p);

    int recv_cnts[p];

    MPI_Alltoall(send_counts, 1, MPI_INT, recv_cnts, 1, MPI_INT, MPI_COMM_WORLD);
//    std::cout<< rank << " recv counts ";
//    printv(recv_cnts, p);
    // ok, now we have the count, let's share data
    // must resize to fit in
    // collect all from others
    int total_count = 0;
    int recv_offset[p];
    for (int i = 0; i < p; i++) {
        recv_offset[i] = total_count;
        total_count += recv_cnts[i];
    }
    int* recv_buf = (int*) malloc(total_count*sizeof(int));

    MPI_Alltoallv(vec, send_counts, send_offset, MPI_INT, recv_buf, recv_cnts, recv_offset, MPI_INT, MPI_COMM_WORLD);

    // do a local sort of the received data
    std::sort(recv_buf, recv_buf+total_count);

//    std::cout<< rank << " final recv buffer ";
//    printv(recv_buf, total_count);

    if(rank == 0)
        printf("N: %d \t The running time is: %f\n", N, MPI_Wtime() - tt);
    MPI_Barrier(MPI_COMM_WORLD);
    // every process writes its result to a file

    { // Write output to a file
        FILE* fd = NULL;
        std::string fname = "ssortoutput-N";
        fname += std::to_string(N);
        fname += ".txt";
        char filename[256];
        snprintf(filename, 256, fname.c_str(), rank);
        fd = fopen(filename,"w+");

        if(NULL == fd) {
            printf("Error opening file \n");
            return 1;
        }

        for(int i = 0; i < N; ++i)
            fprintf(fd, "  %d\n", recv_buf[i]);

        fclose(fd);
    }

    free(vec);
    free(recv_buf);
    free(splitter);
    free(global_splitter);
    MPI_Finalize();
    return 0;
}
