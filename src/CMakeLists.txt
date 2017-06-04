# src
file(GLOB SRC *.c layers/*.c handlers/*.c activations/*.c)
add_library(espresso ${SRC})

if (CUDA_FOUND)
  file(GLOB CU_SRC *.cu layers/*.cu kernels/*.cu activations/*.cu tens.c)
  cuda_add_library(cuespresso ${CU_SRC})
endif()
