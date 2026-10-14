/* Author: abdkhan */
/* KIC gate for microcontrollers: a faithful C port of src/certify/kic.py
 * (KeyedRademacher, EBConfSeq with the Theorem-3 moments and the near-clip
 * companion UpperMeanCS, CertificateBank for the decision target, OnlineCV,
 * ExoChangeDetector, KICGate in 'hold' mode -- every executed move licensed
 * by Proposition 2 -- or 'trust' mode, alpha-spending over certificates
 * (Theorem 4) and the interval-consistency monitor (Proposition 5)) and of
 * the per-tick sequence in loop.run(). The default build is the main
 * gate (loop.py 'kic_sound'); -DKIC_PER_CERT builds 'kic_hold'.
 * Static memory only; no malloc. Double precision, to match the reference. */
#ifndef KIC_GATE_H
#define KIC_GATE_H
#include <stdint.h>

#define KIC_NACT 2          /* duty cycle, transmit power */
#define KIC_NEXO 3          /* interference, link quality, arrival rate */
#define KIC_NFEAT 12        /* 3 exo + 3 squares + 3 cross + 2 centre + 1 */
#ifndef KIC_MAX_ANCHORS
#define KIC_MAX_ANCHORS 24
#endif
#define KIC_MAX_PENDING 16
#define KIC_DET_RECENT 25

typedef struct {
    double alpha, c, logterm;
    int32_t t, nclip;
    double sum_x, sum_sq, mu_hat, sig2_hat, sum_lam, sum_lam_x, sum_v_psi, lo, hi;
    double box_lo[KIC_NACT], box_hi[KIC_NACT];
    /* Theorem 3: lambda-weighted age / location moments */
    double sum_lam_tau, sum_lam_c[KIC_NACT], sum_lam_c2, loc0[KIC_NACT];
    /* near-clip companion (UpperMeanCS, same lambdas) */
    double nz_log, nz_sum_z, nz_mu, nz_sum_lam, nz_sum_lam_z, nz_sum_v_psi;
    int32_t nz_n;
    uint8_t used, has_box, has_loc;
} kic_cs_t;

typedef struct {
    int16_t sign;           /* +1/-1 certified sign of d cost / d actuator, 0 uncertified */
    double lo, hi, extent;
    int32_t n;
    double bias, eta, age;   /* Theorem 3 widening terms (robust mode) */
} kic_cert_t;

typedef struct {
    /* configuration */
    double delta, alpha, bound, radius, L_hat, clip_guard;
    double G, V, near_share;
    uint8_t robust, hold;
    /* Theorem 4: the k-th certificate runs at alpha * 6 / (pi^2 k^2);
     * Proposition 5: reset when current-slope intervals are inconsistent */
    uint8_t family_wise, monitor;
    int32_t k_created, n_alarms;
    int32_t tick;
    int32_t window;
    double max_step, cautious_step, cert_cap, fallback_step, trust_radius;
    uint8_t key[32]; uint32_t key_len;
    /* certificate bank (decision target only) */
    int32_t n_anchors;
    double anchors[KIC_MAX_ANCHORS][KIC_NACT];
    kic_cs_t cs[KIC_MAX_ANCHORS][KIC_NACT];
    kic_cs_t shadow[KIC_MAX_ANCHORS][KIC_NACT];
    double sw[KIC_MAX_ANCHORS][KIC_NACT];
    int32_t sw_n[KIC_MAX_ANCHORS][KIC_NACT];
    int32_t anchor_overflow;
    /* online control variate */
    double xtx[KIC_NFEAT][KIC_NFEAT], xty[KIC_NFEAT], coef[KIC_NFEAT];
    int32_t cv_n; uint8_t cv_has_coef;
    /* exogenous change detector */
    double det_s1[KIC_NEXO], det_s2[KIC_NEXO];
    double det_ring[KIC_DET_RECENT][KIC_NEXO];
    int32_t det_n, det_head;
    /* earned trust */
    double budget;
    int32_t n_pending;
    int8_t pend_a[KIC_MAX_PENDING], pend_d[KIC_MAX_PENDING];
    /* state */
    double nominal[KIC_NACT];
    int32_t n_resets;
} kic_state_t;

void kic_init(kic_state_t *g, const char *key, const double nominal0[KIC_NACT]);
/* One control tick. prop_a: actuator index, -1 = no/unparseable proposal,
 * -2 = non-actionable target; prop_v: proposed value. exo: measured
 * exogenous drivers; cost: measured operator cost J of this tick's output.
 * Outputs the nominal action decided for this tick and the applied action.
 * The measurement (exo, cost) must be the plant's response to *this* tick's
 * applied action, so the caller supplies it after actuation: kic_decide()
 * then kic_observe(). */
void kic_decide(kic_state_t *g, uint32_t t, int prop_a, double prop_v,
                double nominal_out[KIC_NACT], double applied_out[KIC_NACT]);
void kic_observe(kic_state_t *g, uint32_t t, const double exo[KIC_NEXO], double cost);
void kic_resync(kic_state_t *g, const double nominal[KIC_NACT]);

/* SHA-256 / HMAC (used by the keyed PRF) */
void kic_hmac_sha256(const uint8_t *key, uint32_t klen, const uint8_t *msg, uint32_t mlen,
                     uint8_t out[32]);
int kic_prf_sign(const kic_state_t *g, uint32_t t, int j);
#endif
