#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

// gcc mmult3.c -o mmult3.bin -O3 -pthread -ffast-math -march=native

int32_t numthreads;
typedef struct {
  int32_t M;
  int32_t K;
  int32_t N;
  float* Adata;
  float* BdataT;
  float* Cdata;
} mmultfp32t_t;

void randmfp32(int32_t M, int32_t N, float* Adata) {
  // A is MxN (M rows, N columns)
  int32_t i, j;
  for (i=0; i<M; i++) {
    for (j=0; j<N; j++) {
      //Adata[i*N + j] = rand() % 10;
      Adata[i*N + j] = (float)(i+j);
    }
  }
  return;
}
float* newmfp32(int32_t M, int32_t N) {
  // A is MxN (M rows, N columns)
  return (float*)malloc(M*N*sizeof(float));
}

void printmfp32(int32_t M, int32_t N, float* Adata) {
  // A is MxN (M rows, N columns)
  int32_t i, j;
  for (i=0; i<M; i++) {
    for (j=0; j<N; j++) {
      printf("%f ", Adata[i*N + j]);
    }
    printf("\n");
  }
}
void printsummfp32(int32_t M, int32_t N, float* Adata) {
  // A is MxN (M rows, N columns)
  double temp = 0;
  int32_t i, j;
  for (i=0; i<M; i++) {
    for (j=0; j<N; j++) {
      temp += Adata[i*N + j];
    }
  }
  printf("Sum = %lf\n", temp);
}

void mtransposefp32b(int32_t M, int32_t N, float* Adata, float* Cdata) {
  // C = A^T
  int32_t j, k, j2, k2, j2M, blocksize, rblock, cblock, kpcblock;
  blocksize = 64;
  for (j=0; j<N; j+=blocksize) {
    for (k=0; k<M; k+=blocksize) {
      rblock = blocksize;
      cblock = blocksize;
      if ((k + blocksize) >= M) cblock = M - k;
      if ((j + blocksize) >= N) rblock = N - j;
      for (j2=j; j2<j+rblock; j2++) {
        j2M = j2*M;
        kpcblock = k+cblock;
        for (k2=k; k2<kpcblock; k2++) {
          Cdata[j2M + k2] = Adata[k2*N + j2];
        }
      }
    }
  }  
}

void mtransposefp32(int32_t M, int32_t N, float* Adata, float* Cdata) {
  // C = A^T
  int32_t j, k;
  if ((int64_t)M*N >= 10000) return mtransposefp32b(M, N, Adata, Cdata);
  for (j=0; j<N; j++) {
    for (k=0; k<M; k++) {
      Cdata[j*M + k] = Adata[k*N + j];
    }
  }  
}

void *mmultfp32t(void* ptr) {
  mmultfp32t_t *mmultfp32targs = (mmultfp32t_t*)ptr;
  int32_t M = mmultfp32targs->M;
  int32_t K = mmultfp32targs->K;
  int32_t N = mmultfp32targs->N;
  float* Adata = mmultfp32targs->Adata;
  float* BdataT = mmultfp32targs->BdataT;
  float* Cdata = mmultfp32targs->Cdata;
  int32_t i, j, k, iK, jK;
  double temp;
  for (i=0; i<M; i++) {
    iK = i*K;
    for (j=0; j<N; j++) {
      jK = j*K;
      temp = 0;
      for (k=0; k<K; k++) {
        temp += Adata[iK + k] * BdataT[jK + k];
      }
      Cdata[i*N + j] = temp;
    }
  }
}

void mmultfp32(int32_t M, int32_t K, int32_t N, float* Adata, float* Bdata, float* Cdata) {
  // A is MxK (M rows, K columns)
  // B is KXN
  // Calculates C = AB
  if ((M > 65535) || (K > 65535) || (N > 65535)) {
    printf("Matrix dimension too large.\n");
    exit(1);
  };
  float* BdataT = newmfp32(K, N);
  int32_t i, j, k, Arowsperthread, Atrailingrows;
  Arowsperthread = M/numthreads;
  //printf("Using %i threads and %i rows per thread.\n", numthreads, Arowsperthread);
  Atrailingrows = M - Arowsperthread*numthreads;
  mtransposefp32b(K, N, Bdata, BdataT);
  pthread_t *threadid = malloc(numthreads*sizeof(pthread_t));
  mmultfp32t_t *mmultfp32targs = malloc(numthreads*sizeof(mmultfp32t_t));
  if (Arowsperthread > 0) {
   for (i=0; i<numthreads; i++) {
    mmultfp32targs[i].M = Arowsperthread;
    mmultfp32targs[i].K = K;
    mmultfp32targs[i].N = N;
    mmultfp32targs[i].Adata = Adata + i*Arowsperthread*K;
    mmultfp32targs[i].BdataT = BdataT;
    mmultfp32targs[i].Cdata = Cdata + i*Arowsperthread*N;
    pthread_create( &threadid[i], NULL, mmultfp32t, (void*) &mmultfp32targs[i]);
    //mmultfp32t(&mmultfp32targs[i]);
   }
   for (i=0; i<numthreads; i++) {
    pthread_join(threadid[i], NULL);
   }
  }
  if (Atrailingrows > 0) {
    mmultfp32targs[0].M = Atrailingrows;
    mmultfp32targs[0].K = K;
    mmultfp32targs[0].N = N;
    mmultfp32targs[0].Adata = Adata + K*(M - Atrailingrows);
    mmultfp32targs[0].BdataT = BdataT;
    mmultfp32targs[0].Cdata = Cdata + N*(M - Atrailingrows);    
    //mmultfp32t(&mmultfp32targs[0]);
    pthread_create( &threadid[0], NULL, mmultfp32t, (void*) &mmultfp32targs[0]);
    pthread_join(threadid[0], NULL);
  };
  free(BdataT);
}

#define MMULTWINDOWSIZE 32
void mmultfp32window(int32_t M, int32_t K, int32_t N, float* Adata, float* Bdata, float* Cdata) {
  // A is MxK (M rows, K columns)
  // B is KXN
  // Calculates C = AB
  float* BdataT = newmfp32(K, N);
  int32_t ii, jj, i, j, k, windowsize, winrows, wincols;
  mtransposefp32b(K, N, Bdata, BdataT);
  windowsize = MMULTWINDOWSIZE;
  float wincolA[MMULTWINDOWSIZE];
  float wincolBT[MMULTWINDOWSIZE];
  float Sum[MMULTWINDOWSIZE][MMULTWINDOWSIZE] = {{0.0}};
  for (i=0; i<M; i+=windowsize) {
    for (j=0; j<N; j+=windowsize) {
      winrows = windowsize;
      wincols = windowsize;
      if (j+wincols > N) wincols = N - j;
      if (i+winrows > M) winrows = M - i;
      for (k=0; k<K; k++) {
        for (ii=0; ii<winrows; ii++) {
          wincolA[ii] = Adata[(i+ii)*K + k];
        }
        for (jj=0; jj<wincols; jj++) {
          wincolBT[jj] = BdataT[(j+jj)*K + k];
        }        
        for (ii=0; ii<winrows; ii++) {
          for (jj=0; jj<wincols; jj++) {
            Sum[ii][jj] += wincolA[ii]*wincolBT[jj];
          }
        }
      }
      for (ii=0; ii<winrows; ii++) {
        for (jj=0; jj<wincols; jj++) {
          Cdata[(i+ii)*N + j + jj] = Sum[ii][jj];
          Sum[ii][jj] = 0.0;
        }
      }
    }
  }  
}

void main(int32_t argc, char* argv[]) {

  if (argc == 1) {
    printf("Usage: %s M K N\n", argv[0]);
    printf("Multiply a M row K column matrix of floats with a K row N column matrix.\n");
    exit(0);
  }
  uint32_t M,N,K;
  numthreads = 4;
  M = 3;
  K = 4;
  N = 5;
  if (argc == 4) {
    M = atoi(argv[1]);
    K = atoi(argv[2]);
    N = atoi(argv[3]);
  }
  if ((M > 65535) || (K > 65535) || (N > 65535)) {
    printf("Matrix dimension too large.\n");
    exit(1);
  };
  if ((M <= 0) || (K <= 0) || (N <= 0)) {
    printf("Matrix dimension can't be negative or zero.\n");
    exit(1);
  };
  float* Adata = newmfp32(M, K);
  randmfp32(M,K, Adata);
  float* Bdata = newmfp32(K, N);
  randmfp32(K,N, Bdata);
  float* Cdata = newmfp32(M, N);
  uint64_t duration, starttime = time(0);
  mmultfp32(M, K, N, Adata, Bdata, Cdata);
  uint64_t endtime = time(0);
  printsummfp32(M, N, Cdata);
  printf("M*K*N = %i*%i*%i* = %li \n",M, K, N, 1L*M*K*N);
  duration = endtime - starttime;
  printf("Duration = %li s (%i Threads)\n", duration, numthreads);
  if (duration > 0) printf("%f GMKN/s\n", 1L*M*K*N/(1000000000.0*duration));  
  starttime = time(0);
  mmultfp32window(M, K, N, Adata, Bdata, Cdata);
  endtime = time(0);
  printsummfp32(M, N, Cdata);
  printf("M*K*N = %i*%i*%i* = %li \n",M, K, N, 1L*M*K*N);
  duration = endtime - starttime;
  printf("Duration = %li s (1 Thread)\n", duration);
  if (duration > 0) printf("%f GMKN/s\n", 1L*M*K*N/(1000000000.0*duration));
  //printmfp32(M, K, Adata);
  //printmfp32(K, N, Bdata);
  //printmfp32(M, N, Cdata);
}

