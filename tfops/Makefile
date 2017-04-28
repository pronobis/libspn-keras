CC = g++
NVCC = /usr/local/cuda-8.0/bin/nvcc
CXXFLAGS = -std=c++11 -fPIC -lcudart -D GOOGLE_CUDA=1 -D_GLIBCXX_USE_CXX11_ABI=0 -O2
CUDAFLAGS = -std=c++11 -x=cu -Xcompiler -fPIC -D GOOGLE_CUDA=1 -D_GLIBCXX_USE_CXX11_ABI=0
CXXLIBS = -lcudart
LIBDIRS = -L /usr/local/cuda-8.0/lib64/
TF_INCDIR = -I $(shell (python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())'))

OBJ_DIR=obj
_OBJ = gather_columns_functor_gpu.cu.o scatter_columns_functor_gpu.cu.o
OBJ = $(patsubst %,$(OBJ_DIR)/%,$(_OBJ))
OBJ_DEPS = gather_columns_functor_gpu.cu.cc scatter_columns_functor_gpu.cu.cc
SHROBJ_DEP = gather_columns.cc scatter_columns.cc

all: libspn_ops.so

debug: CXXFLAGS += -DEXEC_TIME_CALC
debug: CUDAFLAGS += -DEXEC_TIME_CALC
debug: all

$(OBJ_DIR)/%.o: %.cc $(OBJ_DEPS)
	@mkdir -p $(@D)
	$(NVCC) -c -o $@ $< $(CUDAFLAGS) $(TF_INCDIR)

libspn_ops.so: $(OBJ) $(SHROBJ_DEP)
	$(CC) $(CXXFLAGS) -shared -o $@ $^ $(TF_INCDIR) $(LIBDIRS)

clean:
	rm -rf $(OBJ_DIR)/*.o *.so

