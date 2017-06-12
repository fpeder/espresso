#include "cuesp.h"


int main(int argc, char *argv[])
{
     cufmem_alloc(BYTES(float, 4096*28*28));
     cupmem_alloc(BYTES(uint64_t, 4096*28*28/64));

     mlp tmp = mlp_load("mnist_params", 1);
     cupmlp nn  = cupmlp_convert(&tmp);
     mlp_free(&tmp);

     // batch size = 1
     ftens img = ftens_init(1, 28, 28, 1);
     ftens lab = ftens_init(1, 1,  10, 1);
     mnist_load_X("mnist_test", 0, 1, &img);
     mnist_load_y("mnist_lab", 0, 1, &lab);

     // forward of 1 image
     cuinputLayer  *il = &nn.il;
     cupdenseLayer *dl = nn.dl;
     cubnormLayer *bnl = nn.bnl;

     cuinputLayer_forward(&img, il, 0);

     cupdenseLayer_forward_initial(&il->out, dl, 128);
     cubnormLayer_forward(&dl->out, bnl, 0);
     cupsignAct_forward(&dl->out, &dl->pout);

     dl++; bnl++;
     cupdenseLayer_forward(&(dl-1)->pout, dl, 0);
     cubnormLayer_forward(&dl->out, bnl, 0);
     cupsignAct_forward(&dl->out, &dl->pout);

     dl++; bnl++;
     cupdenseLayer_forward(&(dl-1)->pout, dl, 0);
     cubnormLayer_forward(&dl->out, bnl, 0);
     cupsignAct_forward(&dl->out, &dl->pout);

     dl++; bnl++;
     cupdenseLayer_forward(&(dl-1)->pout, dl, 0);
     cubnormLayer_forward(&dl->out, bnl, 0);

     cudaDeviceSynchronize();
     cuftens_print(&dl->out);

     cupmlp_free(&nn);
     cufmem_free();
     cupmem_free();

     return 0;
}
