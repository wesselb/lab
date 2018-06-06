# Set path for TensorFlow.
TF_INC=-I/usr/local/lib/python3.6/site-packages/tensorflow/include
ifeq ($(shell uname -s),Darwin)
	TF_INC+=-undefined dynamic_lookup
endif

# Set path for Fortan.
F_LIB=-L/usr/local/gfortran/lib -lgfortran

# Configure compilers.
CPP=g++ -O2 -std=c++11 -fPIC
F=gfortran -O2 -fPIC

bvn_cdf_op.o: bvn_cdf_op.cpp tvpack.o
	$(CPP) $(F_LIB) $(TF_INC) -shared -o bvn_cdf_op.so bvn_cdf_op.cpp tvpack.o

tvpack.o: tvpack.f
	$(F) -c tvpack.f




