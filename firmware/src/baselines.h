/* Author: abdkhan */
/* C ports of the gates KIC is compared against (src/certify/loop.py,
 * src/verifier/verify.py), for processor-in-the-loop measurement:
 *   none          execute the proposal, clipped to the actuator range
 *   gstar_target  claim-free graph-sign rule on the operator cost
 *   gstar_claims  the published claim-driven verifier (CausalVerifier.verify)
 *   obs (sign/CI) local-adjusted observational estimator on log + live data */
#ifndef KIC_BASELINES_H
#define KIC_BASELINES_H
#include <stdint.h>

void gate_none(int pa, double pv, const double cur[2], double nxt[2]);
void gate_gstar_target(int pa, double pv, const double cur[2], const int8_t cost_sign[2],
                       double nxt[2]);
/* implied[a][q]: majority-path sign of actuator a on QoS q (latency,
 * throughput, packet_loss, energy); 0 = no path / tie */
void gate_gstar_claims(const char *text, const double cur[2], const int8_t implied[2][4],
                       double nxt[2]);

typedef struct {
    const double (*hist)[6];   /* duty, tx, cost, interference, link, arrival (flash) */
    int32_t n_hist;
    double (*live)[6];         /* ring buffer (RAM) */
    int32_t window, n_live, head;
    double ci_z;               /* <= 0: sign only; else one-sided bound (CIVeX-like) */
} obs_gate_t;
void obs_init(obs_gate_t *g, const double (*hist)[6], int32_t n_hist, double (*live)[6],
              int32_t window, double ci_z);
void obs_decide(obs_gate_t *g, int pa, double pv, const double cur[2], double nxt[2]);
void obs_observe(obs_gate_t *g, const double applied[2], const double exo[3], double cost);
#endif
