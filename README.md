# MxK-times-KxN-32bit-float-matrix-multiply
Written as an excercise in multi-threading, this code provides single and multi-threaded matrix multiplication functions in c for non-square matrices of 32 bit floats. The multi-threading works best when the matrices are not so far from square. It could be improved to handle more extreme non-square examples. The code can be modified to handle matrices of 64 bit doubles or integers as required. Later added a single threaded 'sliding window' algorithm which make more efficient use of memory cache. This method is not yet used in the multi-threaded function.
