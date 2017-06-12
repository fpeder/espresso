# test
set(ESP_LIBS espresso m ${BLAS_LIBRARIES})
set(CUESP_LIBS cuespresso ${CUDA_LIBRARIES})

add_executable(t1 t1.c)
target_link_libraries(t1 ${ESP_LIBS})

add_executable(mnist mnist.c)
target_link_libraries(mnist ${ESP_LIBS})

add_executable(cifar cifar.c)
target_link_libraries(cifar ${ESP_LIBS})

if (CUDA_FOUND)
  cuda_add_executable(cut1 t1.cu)
  target_link_libraries(cut1 ${ESP_LIBS} ${CUESP_LIBS})

  cuda_add_executable(cup_mnist cup_mnist.cu)
  target_link_libraries(cup_mnist ${ESP_LIBS} ${CUESP_LIBS})
endif ()
