/* Author: abdkhan */
/* Minimal Cortex-M startup: vector table, .data copy, .bss zero, main(). */
#include <stdint.h>
extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;
extern int main(void);
void Reset_Handler(void);
void Default_Handler(void) { for (;;) {} }
__attribute__((section(".isr_vector"), used))
const void *vector_table[16] = {
    (void *)&_estack, (void *)Reset_Handler,
    (void *)Default_Handler, (void *)Default_Handler, (void *)Default_Handler,
    (void *)Default_Handler, (void *)Default_Handler, 0, 0, 0, 0,
    (void *)Default_Handler, (void *)Default_Handler, 0,
    (void *)Default_Handler, (void *)Default_Handler};
void Reset_Handler(void) {
    uint32_t *s = &_sidata, *d = &_sdata;
    while (d < &_edata) *d++ = *s++;
    for (d = &_sbss; d < &_ebss;) *d++ = 0;
#if defined(__ARM_FP)
    /* enable the FPU (CP10/CP11 full access) on M4F */
    *(volatile uint32_t *)0xE000ED88 |= (0xFu << 20);
    __asm volatile("dsb; isb");
#endif
    main();
    for (;;) {}
}
