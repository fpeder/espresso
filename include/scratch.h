#ifndef ESP_SCRATCH_H
#define ESP_SCRATCH_H

void scratch_alloc(int len);
void scratch_free();
void dfscratch_alloc(int len);
void dfscratch_free();
void dpscratch_alloc(int len);
void dpscratch_free();

#endif /* ESP_SCRATCH_H */
