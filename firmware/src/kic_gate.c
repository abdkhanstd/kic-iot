/* Author: abdkhan */
/* C port of the KIC gate; see kic_gate.h. Each function names the Python
 * code it mirrors so the two can be checked line by line. */
#include "kic_gate.h"
#include <math.h>
#include <string.h>

#define LO 0.02
#define HI 0.98
static double last_centre[KIC_NACT];
static int last_signs[KIC_NACT];

static double clipd(double x, double a, double b) { return x < a ? a : (x > b ? b : x); }

/* ------------------------------------------------------------ SHA-256 */
static const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
#define ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
typedef struct { uint32_t h[8]; uint8_t buf[64]; uint32_t blen; uint64_t len; } sha_t;
static void sha_block(sha_t *s, const uint8_t *p) {
    uint32_t w[64], a, b, c, d, e, f, g, h, t1, t2; int i;
    for (i = 0; i < 16; i++) w[i] = (uint32_t)p[4*i] << 24 | (uint32_t)p[4*i+1] << 16 | (uint32_t)p[4*i+2] << 8 | p[4*i+3];
    for (i = 16; i < 64; i++) {
        uint32_t s0 = ROR(w[i-15], 7) ^ ROR(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = ROR(w[i-2], 17) ^ ROR(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    a = s->h[0]; b = s->h[1]; c = s->h[2]; d = s->h[3]; e = s->h[4]; f = s->h[5]; g = s->h[6]; h = s->h[7];
    for (i = 0; i < 64; i++) {
        t1 = h + (ROR(e, 6) ^ ROR(e, 11) ^ ROR(e, 25)) + ((e & f) ^ (~e & g)) + K256[i] + w[i];
        t2 = (ROR(a, 2) ^ ROR(a, 13) ^ ROR(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    s->h[0] += a; s->h[1] += b; s->h[2] += c; s->h[3] += d; s->h[4] += e; s->h[5] += f; s->h[6] += g; s->h[7] += h;
}
static void sha_init(sha_t *s) {
    static const uint32_t iv[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    memcpy(s->h, iv, sizeof iv); s->blen = 0; s->len = 0;
}
static void sha_update(sha_t *s, const uint8_t *p, uint32_t n) {
    while (n--) { s->buf[s->blen++] = *p++; s->len += 8; if (s->blen == 64) { sha_block(s, s->buf); s->blen = 0; } }
}
static void sha_final(sha_t *s, uint8_t out[32]) {
    uint64_t L = s->len; uint8_t pad = 0x80, z = 0; int i;
    sha_update(s, &pad, 1); s->len -= 8;
    while (s->blen != 56) { sha_update(s, &z, 1); s->len -= 8; }
    for (i = 7; i >= 0; i--) { uint8_t b = (uint8_t)(L >> (8 * i)); sha_update(s, &b, 1); }
    for (i = 0; i < 8; i++) { out[4*i] = s->h[i] >> 24; out[4*i+1] = s->h[i] >> 16; out[4*i+2] = s->h[i] >> 8; out[4*i+3] = s->h[i]; }
}
void kic_hmac_sha256(const uint8_t *key, uint32_t klen, const uint8_t *msg, uint32_t mlen, uint8_t out[32]) {
    uint8_t k0[64], ip[64], op[64], inner[32]; sha_t s; int i;
    memset(k0, 0, 64);
    if (klen > 64) { sha_init(&s); sha_update(&s, key, klen); sha_final(&s, k0); } else memcpy(k0, key, klen);
    for (i = 0; i < 64; i++) { ip[i] = k0[i] ^ 0x36; op[i] = k0[i] ^ 0x5c; }
    sha_init(&s); sha_update(&s, ip, 64); sha_update(&s, msg, mlen); sha_final(&s, inner);
    sha_init(&s); sha_update(&s, op, 64); sha_update(&s, inner, 32); sha_final(&s, out);
}
/* KeyedRademacher.sign: HMAC-SHA256(K, f"{t}:{j}")[0] & 1 -> +1 else -1 */
int kic_prf_sign(const kic_state_t *g, uint32_t t, int j) {
    char msg[24]; int n = 0; char tmp[12]; int k = 0; uint8_t d[32];
    uint32_t v = t;
    do { tmp[k++] = (char)('0' + v % 10); v /= 10; } while (v);
    while (k) msg[n++] = tmp[--k];
    msg[n++] = ':';
    msg[n++] = (char)('0' + j);
    kic_hmac_sha256(g->key, g->key_len, (const uint8_t *)msg, (uint32_t)n, d);
    return (d[0] & 1) ? 1 : -1;
}

/* ------------------------------------------------------------ EBConfSeq */
/* EBConfSeq(alpha*(1-near_share)) + UpperMeanCS(alpha*near_share) in robust mode */
static void cs_init(kic_state_t *g, kic_cs_t *c, double alpha) {
    memset(c, 0, sizeof *c);
    if (g->family_wise) {   /* CertificateBank._new_cs: alpha_k = alpha * 6 / (pi^2 k^2) */
        g->k_created += 1;
        alpha *= 6.0 / (9.869604401089358 * (double)g->k_created * (double)g->k_created);
    }
    if (g->robust) {
        c->nz_log = log(1.0 / (alpha * g->near_share));
        alpha *= 1.0 - g->near_share;
    }
    c->alpha = alpha; c->c = 0.75; c->logterm = log(2.0 / alpha);
    c->mu_hat = 0.5; c->sig2_hat = 0.25; c->lo = 0.0; c->hi = 1.0; c->used = 1;
}
static double psi_e(double lam) { return (-log(1.0 - lam) - lam) / 4.0; }
/* UpperMeanCS.upper() */
static double nz_upper(const kic_cs_t *c) {
    double u;
    if (c->nz_sum_lam == 0.0) return 1.0;
    u = (c->nz_sum_lam_z + c->nz_log + c->nz_sum_v_psi) / c->nz_sum_lam;
    return u < 1.0 ? u : 1.0;
}
/* EBConfSeq.rms_dist() */
static double cs_rms(const kic_cs_t *c, const double x[KIC_NACT]) {
    double m2, xs = 0.0, xx = 0.0; int b;
    if (c->sum_lam == 0.0 || !c->has_loc) return 0.0;
    for (b = 0; b < KIC_NACT; b++) {
        double z = x[b] - c->loc0[b];
        xs += z * c->sum_lam_c[b]; xx += z * z;
    }
    m2 = (c->sum_lam_c2 - 2.0 * xs + xx * c->sum_lam) / c->sum_lam;
    return sqrt(m2 > 0.0 ? m2 : 0.0);
}
static void cs_update(kic_cs_t *c, double x, int32_t tau, const double loc[KIC_NACT], int near) {
    double lam, v, centre, width, psi;
    int b;
    x = clipd(x, 0.0, 1.0);
    c->t += 1;
    lam = sqrt(2.0 * c->logterm / (c->sig2_hat * c->t * log(1.0 + c->t)));
    if (lam > c->c) lam = c->c;
    if (loc) {   /* moments about the first location (no cancellation) */
        double xx = 0.0;
        if (!c->has_loc) for (b = 0; b < KIC_NACT; b++) c->loc0[b] = loc[b];
        c->sum_lam_tau += lam * tau;
        for (b = 0; b < KIC_NACT; b++) {
            double z = loc[b] - c->loc0[b];
            c->sum_lam_c[b] += lam * z; xx += z * z;
        }
        c->sum_lam_c2 += lam * xx;
        c->has_loc = 1;
    }
    if (near >= 0) {   /* UpperMeanCS.update(z, lam) */
        double vz = 4.0 * (near - c->nz_mu) * (near - c->nz_mu);
        c->nz_sum_lam += lam; c->nz_sum_lam_z += lam * near; c->nz_sum_v_psi += vz * psi_e(lam);
        c->nz_n += 1; c->nz_sum_z += near;
        c->nz_mu = (0.5 + c->nz_sum_z) / (c->nz_n + 1);
    }
    v = 4.0 * (x - c->mu_hat) * (x - c->mu_hat);
    psi = psi_e(lam);
    c->sum_lam += lam; c->sum_lam_x += lam * x; c->sum_v_psi += v * psi;
    c->sum_x += x;
    c->sum_sq += (x - c->mu_hat) * (x - c->mu_hat);
    c->mu_hat = (0.5 + c->sum_x) / (c->t + 1);
    c->sig2_hat = (0.25 + c->sum_sq) / (c->t + 1);
    centre = c->sum_lam_x / c->sum_lam;
    width = (c->logterm + c->sum_v_psi) / c->sum_lam;
    c->lo = centre - width; if (c->lo < 0.0) c->lo = 0.0;
    c->hi = centre + width; if (c->hi > 1.0) c->hi = 1.0;
}
static void cs_visit(kic_cs_t *c, const double x[KIC_NACT], int clipped) {
    int b;
    c->nclip += clipped;
    for (b = 0; b < KIC_NACT; b++) {
        if (!c->has_box) { c->box_lo[b] = x[b]; c->box_hi[b] = x[b]; }
        else { if (x[b] < c->box_lo[b]) c->box_lo[b] = x[b]; if (x[b] > c->box_hi[b]) c->box_hi[b] = x[b]; }
    }
    c->has_box = 1;
}

/* ------------------------------------------------------------ CertificateBank */
static void bank_reset(kic_state_t *g) {
    g->n_anchors = 0;
    memset(g->cs, 0, sizeof g->cs); memset(g->shadow, 0, sizeof g->shadow);
    memset(g->sw, 0, sizeof g->sw); memset(g->sw_n, 0, sizeof g->sw_n);
}
/* CertificateBank.cell(): nearest anchor within the L-inf radius, else -1 */
static int bank_cell(kic_state_t *g, const double x[KIC_NACT], int create) {
    int i, b, best = -1; double bd = 0.0;
    for (i = 0; i < g->n_anchors; i++) {
        double d = 0.0;
        for (b = 0; b < KIC_NACT; b++) { double e = fabs(g->anchors[i][b] - x[b]); if (e > d) d = e; }
        if (d <= g->radius + 1e-12 && (best < 0 || d < bd)) { best = i; bd = d; }
    }
    if (best < 0 && create) {
        if (g->n_anchors >= KIC_MAX_ANCHORS) { g->anchor_overflow++; return -1; }
        for (b = 0; b < KIC_NACT; b++) g->anchors[g->n_anchors][b] = x[b];
        best = g->n_anchors++;
    }
    return best;
}
static double beta_hat(const kic_state_t *g, int cell, int a) {
    return g->sw_n[cell][a] ? g->sw[cell][a] / (g->sw_n[cell][a] * g->delta) : 0.0;
}
/* CertificateBank.update() for the decision target */
static void bank_update(kic_state_t *g, const double centre[KIC_NACT], const int signs[KIC_NACT],
                        double y, double yhat) {
    int a, b, cell = bank_cell(g, centre, 1);
    double r0 = y - yhat, bh[KIC_NACT];
    g->tick += 1;
    if (cell < 0) return;
    for (b = 0; b < KIC_NACT; b++) bh[b] = beta_hat(g, cell, b);   /* frozen: past ticks only */
    for (a = 0; a < KIC_NACT; a++) {
        double other = 0.0, r, w, x; int clipped, near;
        kic_cs_t *main;
        for (b = 0; b < KIC_NACT; b++) if (b != a) other += g->delta * signs[b] * bh[b];
        r = r0 - other;
        clipped = fabs(r) > g->bound;
        w = signs[a] * clipd(r, -g->bound, g->bound) / g->bound;
        x = (w + 1.0) / 2.0;
        near = g->robust ? (fabs(r) > g->bound - 2.0 * g->delta * g->G) : -1;
        main = &g->cs[cell][a];
        if (!main->used) cs_init(g, main, g->alpha);
        cs_update(main, x, g->tick, centre, near); cs_visit(main, centre, clipped);
        if (g->window && main->t >= g->window) {
            kic_cs_t *sh = &g->shadow[cell][a];
            if (!sh->used) cs_init(g, sh, g->alpha);
            cs_update(sh, x, g->tick, centre, near); cs_visit(sh, centre, clipped);
            if (sh->t >= g->window) { *main = *sh; memset(sh, 0, sizeof *sh); }
        }
        g->sw[cell][a] += signs[a] * r; g->sw_n[cell][a] += 1;
    }
}
/* CertificateBank.query() */
static kic_cert_t bank_query(kic_state_t *g, int a, const double centre[KIC_NACT]) {
    kic_cert_t q = {0, -INFINITY, INFINITY, 0.0, 0, 0.0, 0.0, 0.0};
    int b, cell = bank_cell(g, centre, 0);
    kic_cs_t *c;
    if (cell < 0) return q;
    c = &g->cs[cell][a];
    if (!c->used) return q;
    q.lo = (2 * c->lo - 1) * g->bound / g->delta;
    q.hi = (2 * c->hi - 1) * g->bound / g->delta;
    q.sign = q.lo > 0 ? 1 : (q.hi < 0 ? -1 : 0);
    q.n = c->t;
    if (c->has_box)
        for (b = 0; b < KIC_NACT; b++) {
            double e1 = fabs(centre[b] - c->box_lo[b]), e2 = fabs(centre[b] - c->box_hi[b]);
            double e = e1 > e2 ? e1 : e2; if (e > q.extent) q.extent = e;
        }
    if (g->robust) {
        q.bias = g->G * nz_upper(c);
        q.eta = cs_rms(c, centre);
        q.age = c->sum_lam > 0.0 ? g->tick - c->sum_lam_tau / c->sum_lam : 0.0;
    } else {
        double lim = g->clip_guard * c->t; if (lim < 1.0) lim = 1.0;
        if (c->nclip > lim) q.sign = 0;
    }
    return q;
}

/* ------------------------------------------------------------ OnlineCV (decision target) */
static void cv_feat(const double e[KIC_NEXO], const double c[KIC_NACT], double f[KIC_NFEAT]) {
    f[0] = e[0]; f[1] = e[1]; f[2] = e[2];
    f[3] = e[0]*e[0]; f[4] = e[1]*e[1]; f[5] = e[2]*e[2];
    f[6] = e[0]*e[1]; f[7] = e[0]*e[2]; f[8] = e[1]*e[2];
    f[9] = c[0]; f[10] = c[1]; f[11] = 1.0;
}
static void cv_solve(kic_state_t *g) {   /* (XtX + ridge I) coef = Xty, Gaussian elimination */
    double A[KIC_NFEAT][KIC_NFEAT + 1]; int i, j, k, p;
    for (i = 0; i < KIC_NFEAT; i++) {
        for (j = 0; j < KIC_NFEAT; j++) A[i][j] = g->xtx[i][j] + (i == j ? 1e-3 : 0.0);
        A[i][KIC_NFEAT] = g->xty[i];
    }
    for (k = 0; k < KIC_NFEAT; k++) {
        p = k;
        for (i = k + 1; i < KIC_NFEAT; i++) if (fabs(A[i][k]) > fabs(A[p][k])) p = i;
        if (p != k) for (j = 0; j <= KIC_NFEAT; j++) { double t = A[k][j]; A[k][j] = A[p][j]; A[p][j] = t; }
        for (i = k + 1; i < KIC_NFEAT; i++) {
            double m = A[i][k] / A[k][k];
            for (j = k; j <= KIC_NFEAT; j++) A[i][j] -= m * A[k][j];
        }
    }
    for (i = KIC_NFEAT - 1; i >= 0; i--) {
        double s = A[i][KIC_NFEAT];
        for (j = i + 1; j < KIC_NFEAT; j++) s -= A[i][j] * g->coef[j];
        g->coef[i] = s / A[i][i];
    }
    g->cv_has_coef = 1;
}
static void cv_update(kic_state_t *g, const double e[KIC_NEXO], const double c[KIC_NACT], double y) {
    double f[KIC_NFEAT]; int i, j;
    cv_feat(e, c, f);
    for (i = 0; i < KIC_NFEAT; i++) { g->xty[i] += f[i] * y; for (j = 0; j < KIC_NFEAT; j++) g->xtx[i][j] += f[i] * f[j]; }
    g->cv_n++;
    if (g->cv_n >= 20 && g->cv_n % 10 == 0) cv_solve(g);
}

/* ------------------------------------------------------------ ExoChangeDetector */
static int det_update(kic_state_t *g, const double e[KIC_NEXO]) {
    int k, i; double zmax = 0.0;
    /* append e: running sums over the whole buffer + ring of the last 25 */
    for (k = 0; k < KIC_NEXO; k++) { g->det_s1[k] += e[k]; g->det_s2[k] += e[k] * e[k]; g->det_ring[g->det_head][k] = e[k]; }
    g->det_head = (g->det_head + 1) % KIC_DET_RECENT;
    g->det_n++;
    if (g->det_n < 40 + KIC_DET_RECENT) return 0;
    for (k = 0; k < KIC_NEXO; k++) {
        double rs = 0.0, nb = g->det_n - KIC_DET_RECENT, bs1, bs2, mb, sd, z;
        double rs2 = 0.0;
        for (i = 0; i < KIC_DET_RECENT; i++) { rs += g->det_ring[i][k]; rs2 += g->det_ring[i][k] * g->det_ring[i][k]; }
        bs1 = g->det_s1[k] - rs; bs2 = g->det_s2[k] - rs2;
        mb = bs1 / nb;
        sd = sqrt(fmax(bs2 / nb - mb * mb, 0.0)) + 1e-6;
        z = fabs(rs / KIC_DET_RECENT - mb) / (sd / sqrt((double)KIC_DET_RECENT));
        if (z > zmax) zmax = z;
    }
    if (zmax > 5.0) {   /* keep only the last 25 readings */
        for (k = 0; k < KIC_NEXO; k++) {
            double rs = 0.0, rs2 = 0.0;
            for (i = 0; i < KIC_DET_RECENT; i++) { rs += g->det_ring[i][k]; rs2 += g->det_ring[i][k] * g->det_ring[i][k]; }
            g->det_s1[k] = rs; g->det_s2[k] = rs2;
        }
        g->det_n = KIC_DET_RECENT;
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------ KICGate (trust mode) */
static double certified_step(const kic_state_t *g, const kic_cert_t *c, int d, double cap) {
    double m, slack;
    if (c->sign == 0 || d * c->sign != -1) return 0.0;     /* sense = -1: minimise J */
    m = d > 0 ? -c->hi : c->lo;
    if (g->robust)   /* Theorem 3: current slope at the current action */
        slack = m - c->bias - g->L_hat * (c->eta + 0.5 * g->delta) - g->V * (c->age + 1.0);
    else
        slack = m - g->L_hat * (c->extent + 0.5 * g->delta);
    if (slack <= 0) return 0.0;
    return fmin(cap, 2.0 * slack / g->L_hat);
}
static void fallback(kic_state_t *g, const double cur[KIC_NACT], double nxt[KIC_NACT]) {
    int a, best = -1; double bstr = 0.0, bv = 0.0;
    for (a = 0; a < KIC_NACT; a++) {
        kic_cert_t c = bank_query(g, a, cur);
        double strength, st, v; int d;
        if (c.sign == 0) continue;
        strength = fmin(fabs(c.lo), fabs(c.hi));
        d = -c.sign;
        st = certified_step(g, &c, d, g->fallback_step);
        if (st <= 0) continue;
        v = clipd(cur[a] + d * st, LO, HI);
        if (fabs(v - cur[a]) < 1e-9) continue;
        if (best < 0 || strength > bstr) { best = a; bstr = strength; bv = v; }
    }
    nxt[0] = cur[0]; nxt[1] = cur[1];
    if (best >= 0) nxt[best] = bv;
}
static void audit(kic_state_t *g, const double cur[KIC_NACT]) {
    int i, n = 0; int8_t ka[KIC_MAX_PENDING], kd[KIC_MAX_PENDING];
    for (i = 0; i < g->n_pending; i++) {
        kic_cert_t c = bank_query(g, g->pend_a[i], cur);
        if (c.sign == 0) { ka[n] = g->pend_a[i]; kd[n] = g->pend_d[i]; n++; }
        else if (g->pend_d[i] * c.sign == -1) g->budget = g->trust_radius;
        else { g->budget = 0.0; n = 0; break; }
    }
    if (n > 8) { memmove(ka, ka + n - 8, 8); memmove(kd, kd + n - 8, 8); n = 8; }
    memcpy(g->pend_a, ka, n); memcpy(g->pend_d, kd, n); g->n_pending = n;
}
static void decide(kic_state_t *g, int pa, double pv, const double cur[KIC_NACT], double nxt[KIC_NACT]) {
    double step; int d; kic_cert_t c;
    nxt[0] = cur[0]; nxt[1] = cur[1];
    if (pa == -1) { fallback(g, cur, nxt); return; }
    if (!g->hold) audit(g, cur);
    if (pa < 0 || pa >= KIC_NACT) { fallback(g, cur, nxt); return; }
    if (!(pv >= 0.0 && pv <= 1.0)) { fallback(g, cur, nxt); return; }
    step = clipd(pv - cur[pa], -g->max_step, g->max_step);
    if (fabs(step) < 1e-9 || fabs(clipd(cur[pa] + step, LO, HI) - cur[pa]) < 1e-9) { fallback(g, cur, nxt); return; }
    d = step > 0 ? 1 : -1;
    c = bank_query(g, pa, cur);
    if (c.sign != 0) {
        if (d * c.sign == -1) {
            double cap = certified_step(g, &c, d, g->cert_cap);
            if (cap > 0) { nxt[pa] = clipd(cur[pa] + clipd(step, -cap, cap), LO, HI); return; }
        }
        fallback(g, cur, nxt); return;
    }
    if (!g->hold && g->budget + 1e-9 >= g->cautious_step) {
        nxt[pa] = clipd(cur[pa] + clipd(step, -g->cautious_step, g->cautious_step), LO, HI);
        g->budget -= fabs(nxt[pa] - cur[pa]);
        if (g->n_pending < KIC_MAX_PENDING) { g->pend_a[g->n_pending] = (int8_t)pa; g->pend_d[g->n_pending] = (int8_t)d; g->n_pending++; }
        return;
    }
    fallback(g, cur, nxt);
}

/* ------------------------------------------------------------ Proposition 5 monitor */
/* CertificateBank.current_slope_intervals(): every live certificate of
 * actuator a (all anchors, main and shadow) gives an interval on the current
 * slope at x; on the coverage event they all contain the same number. */
static int monitor_inconsistent(kic_state_t *g, const double x[KIC_NACT]) {
    int a, i, s;
    for (a = 0; a < KIC_NACT; a++) {
        double maxlo = -INFINITY, minhi = INFINITY; int n = 0;
        for (i = 0; i < g->n_anchors; i++)
            for (s = 0; s < 2; s++) {
                const kic_cs_t *c = s ? &g->shadow[i][a] : &g->cs[i][a];
                double lo, hi, w, age;
                if (!c->used || c->t == 0) continue;
                lo = (2 * c->lo - 1) * g->bound / g->delta;
                hi = (2 * c->hi - 1) * g->bound / g->delta;
                age = c->sum_lam > 0.0 ? g->tick - c->sum_lam_tau / c->sum_lam : 0.0;
                w = g->G * nz_upper(c) + g->L_hat * (cs_rms(c, x) + 0.5 * g->delta) + g->V * (age + 1.0);
                if (lo - w > maxlo) maxlo = lo - w;
                if (hi + w < minhi) minhi = hi + w;
                n++;
            }
        if (n > 1 && maxlo > minhi) return 1;
    }
    return 0;
}

/* ------------------------------------------------------------ public API (loop.run order) */
void kic_init(kic_state_t *g, const char *key, const double nominal0[KIC_NACT]) {
    memset(g, 0, sizeof *g);
    g->delta = 0.05; g->alpha = 0.05; g->radius = 0.1; g->L_hat = 3.5;
#ifndef KIC_PER_CERT
    g->family_wise = 1; g->monitor = 1;
#endif
    g->clip_guard = 0.01; g->window = 400;   /* W = 0.08 / V with V = 2e-4 */
#ifndef KIC_REV2
    g->robust = 1; g->hold = 1; g->bound = 0.35; g->G = 1.6; g->V = 2e-4; g->near_share = 0.2;
#else
    g->bound = 0.2;
#endif
    g->max_step = 0.35; g->cautious_step = 0.05; g->cert_cap = 0.1; g->fallback_step = 0.05;
    g->trust_radius = 0.15; g->budget = 0.15;
    g->key_len = (uint32_t)strlen(key); memcpy(g->key, key, g->key_len);
    g->nominal[0] = nominal0[0]; g->nominal[1] = nominal0[1];
}
void kic_decide(kic_state_t *g, uint32_t t, int pa, double pv, double nom_out[KIC_NACT], double app_out[KIC_NACT]) {
    double nxt[KIC_NACT]; int a;
    if (g->monitor && g->robust && monitor_inconsistent(g, g->nominal)) {
        bank_reset(g);                       /* loop.run: bank.reset(); gate.reset_trust() */
        g->budget = g->trust_radius; g->n_pending = 0;
        g->n_alarms++;
    }
    decide(g, pa, pv, g->nominal, nxt);
    for (a = 0; a < KIC_NACT; a++) nxt[a] = clipd(nxt[a], LO + g->delta, HI - g->delta);
    for (a = 0; a < KIC_NACT; a++) {
        g->nominal[a] = nxt[a];
        last_centre[a] = clipd(nxt[a], LO + g->delta, HI - g->delta);
        last_signs[a] = kic_prf_sign(g, t, a);
        nom_out[a] = nxt[a];
        app_out[a] = last_centre[a] + g->delta * last_signs[a];
    }
}
void kic_resync(kic_state_t *g, const double nominal[KIC_NACT]) {
    int a;
    for (a = 0; a < KIC_NACT; a++) {
        g->nominal[a] = nominal[a];
        last_centre[a] = clipd(nominal[a], LO + g->delta, HI - g->delta);
    }
}
void kic_observe(kic_state_t *g, uint32_t t, const double exo[KIC_NEXO], double cost) {
    double f[KIC_NFEAT], yhat = 0.0; int i, have = g->cv_has_coef;
    (void)t;
    if (have) { cv_feat(exo, last_centre, f); for (i = 0; i < KIC_NFEAT; i++) yhat += f[i] * g->coef[i]; }
    cv_update(g, exo, last_centre, cost);
    if (have) bank_update(g, last_centre, last_signs, cost, yhat);
    if (det_update(g, exo)) {
        bank_reset(g);
        memset(g->xtx, 0, sizeof g->xtx); memset(g->xty, 0, sizeof g->xty);
        g->cv_n = 0; g->cv_has_coef = 0;
        g->budget = g->trust_radius; g->n_pending = 0;
        g->n_resets++;
    }
}
