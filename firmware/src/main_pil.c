/* Author: abdkhan */
/* Processor-in-the-loop replay. The trace (generated from the Python closed
 * loop) supplies, per tick, the parsed proposal and the measured exogenous
 * drivers and cost; the firmware runs the C gate and checks that the nominal
 * and applied actions it produces are the ones the reference produced. If
 * every tick matches, the closed loop with the firmware gate is identical to
 * the reference loop. tick_done() is hooked by the emulator to count
 * instructions per tick. */
#include <stdint.h>
#include <math.h>
#include "kic_gate.h"
#include "baselines.h"
#include "trace.h"

volatile uint32_t pil_mismatch, pil_first_mismatch, pil_ticks, pil_stack_bytes,
    pil_overflow, pil_max_anchors, pil_resets, pil_done;
#ifndef GATE_BASE
static kic_state_t G;
#else
#if GATE_BASE == 3
static obs_gate_t O;
static double LIVE[OBS_WINDOW][6];
#endif
#endif
extern uint32_t _ebss;

__attribute__((noinline)) void tick_done(uint32_t t) { __asm volatile("" ::: "memory"); (void)t; }
/* hooked by the emulator to report the results and stop */
__attribute__((noinline)) void pil_finish(void) { __asm volatile("" ::: "memory"); }

static uint32_t *stack_paint(void) {
    uint32_t sp_here; uint32_t *p = &_ebss, *top = (uint32_t *)(void *)(&sp_here) - 64;
    while (p < top) *p++ = 0xDEADBEEFu;
    return top;
}
static uint32_t stack_used(uint32_t *top) {
    uint32_t *p = &_ebss;
    while (p < top && *p == 0xDEADBEEFu) p++;
    return (uint32_t)((uint8_t *)top - (uint8_t *)p) + 256u;
}

#ifndef GATE_BASE
int main(void) {
    uint32_t *top = stack_paint();
    double nom[KIC_NACT], app[KIC_NACT];
    uint32_t t, mism = 0, first = 0;
    kic_init(&G, TRACE_KEY, TRACE_NOMINAL0);
    tick_done(0);
    for (t = 1; t <= TRACE_N; t++) {
        uint32_t i = t - 1;
        kic_decide(&G, t, TRACE_PA[i], TRACE_PV[i], nom, app);
        if (fabs(nom[0] - TRACE_NOM[i][0]) > 1e-9 || fabs(nom[1] - TRACE_NOM[i][1]) > 1e-9 ||
            fabs(app[0] - TRACE_APP[i][0]) > 1e-9 || fabs(app[1] - TRACE_APP[i][1]) > 1e-9) {
            if (!mism) first = t;
            mism++;
            /* resynchronise to the reference so one borderline decision does
             * not cascade; mismatches are counted, not hidden */
            kic_resync(&G, TRACE_NOM[i]);
        }
        kic_observe(&G, t, TRACE_EXO[i], TRACE_COST[i]);
        if ((uint32_t)G.n_anchors > pil_max_anchors) pil_max_anchors = (uint32_t)G.n_anchors;
        tick_done(t);
    }
    pil_mismatch = mism; pil_first_mismatch = first; pil_ticks = TRACE_N;
    pil_overflow = (uint32_t)G.anchor_overflow; pil_resets = (uint32_t)G.n_resets;
    pil_stack_bytes = stack_used(top);
    pil_done = 0xC0FFEEu;
    pil_finish();
    for (;;) __asm volatile("wfi");   /* sleep: the emulator fast-forwards */
}
#else
/* comparison gates: the only gate state is the current nominal action (and,
 * for the observational gates, the live telemetry window) */
int main(void) {
    uint32_t *top = stack_paint();
    double cur[2] = {TRACE_NOMINAL0[0], TRACE_NOMINAL0[1]}, nxt[2];
    uint32_t i, mism = 0, first = 0;
#if GATE_BASE == 3
    obs_init(&O, HIST, HIST_N, LIVE, OBS_WINDOW, OBS_Z);
#endif
    tick_done(0);
    for (i = 0; i < TRACE_N; i++) {
#if GATE_BASE == 0
        gate_none(TRACE_PA[i], TRACE_PV[i], cur, nxt);
#elif GATE_BASE == 1
        gate_gstar_target(TRACE_PA[i], TRACE_PV[i], cur, GATE_COST_SIGN, nxt);
#elif GATE_BASE == 2
        gate_gstar_claims(TRACE_TEXT + TRACE_TOFF[i], cur, GATE_IMPLIED, nxt);
#else
        obs_decide(&O, TRACE_PA[i], TRACE_PV[i], cur, nxt);
#endif
        if (fabs(nxt[0] - TRACE_NOM[i][0]) > 1e-9 || fabs(nxt[1] - TRACE_NOM[i][1]) > 1e-9) {
            if (!mism) first = TRACE_T0 + i;
            mism++;
            nxt[0] = TRACE_NOM[i][0]; nxt[1] = TRACE_NOM[i][1];
        }
#if GATE_BASE == 3
        obs_observe(&O, nxt, TRACE_EXO[i], TRACE_COST[i]);   /* no dither: applied = nominal */
#endif
        cur[0] = nxt[0]; cur[1] = nxt[1];
        tick_done(TRACE_T0 + i);
    }
    pil_mismatch = mism; pil_first_mismatch = first; pil_ticks = TRACE_N;
    pil_stack_bytes = stack_used(top);
    pil_done = 0xC0FFEEu;
    pil_finish();
    for (;;) __asm volatile("wfi");   /* sleep: the emulator fast-forwards */
}
#endif
