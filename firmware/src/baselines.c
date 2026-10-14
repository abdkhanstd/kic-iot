/* Author: abdkhan */
/* C ports of the comparison gates; see baselines.h. Each function reproduces
 * the Python reference decision for one tick given the same inputs. */
#include <math.h>
#include <string.h>
#include "baselines.h"

#define LO 0.02
#define HI 0.98
#define MAX_STEP 0.35
#define NO_OP_EPS 0.01

static double clipd(double x, double a, double b) { return x < a ? a : (x > b ? b : x); }
static double sgn(double x) { return (double)((x > 0) - (x < 0)); }

/* ------------------------------------------------------------------ none */
void gate_none(int pa, double pv, const double cur[2], double nxt[2]) {
    nxt[0] = cur[0]; nxt[1] = cur[1];
    if (pa >= 0) nxt[pa] = clipd(pv, LO, HI);
}

/* ------------------------------------------ G*, claim-free graph-sign rule */
void gate_gstar_target(int pa, double pv, const double cur[2], const int8_t cost_sign[2],
                       double nxt[2]) {
    nxt[0] = cur[0]; nxt[1] = cur[1];
    if (pa < 0 || !(pv >= 0.0 && pv <= 1.0)) return;
    double c = clipd(cur[pa] + clipd(pv - cur[pa], -MAX_STEP, MAX_STEP), LO, HI);
    if (fabs(c - cur[pa]) < 1e-9) return;
    /* execute only a move the reference graph says lowers the cost */
    if (cost_sign[pa] != 0 && sgn(c - cur[pa]) * cost_sign[pa] == -1) nxt[pa] = c;
}

/* ------------------------------ G*, claim-driven verifier (verify.py) ---- */
static int is_word(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
}
static int is_alpha_(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }
static int is_digit(char c) { return c >= '0' && c <= '9'; }
static int is_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
static char lower(char c) { return (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c; }

/* case-insensitive literal match; '#' in the pattern matches '_' or ' ' */
static int lit_ci(const char *s, int n, int p, const char *pat) {
    int k;
    for (k = 0; pat[k]; k++) {
        if (p + k >= n) return 0;
        char c = lower(s[p + k]);
        if (pat[k] == '#' ? !(c == '_' || c == ' ') : c != pat[k]) return 0;
    }
    return k;
}

/* _DO_ARG: ([a-zA-Z_]+)\s*=\s*(-?[0-9]*\.?[0-9]+), searched in s[p0, p1) */
static int do_arg(const char *s, int p, int p1, int *vs, int *ve, int *ns, int *ne) {
    int q = p;
    if (q >= p1 || !is_alpha_(s[q])) return 0;
    while (q < p1 && is_alpha_(s[q])) q++;
    *vs = p; *ve = q;
    while (q < p1 && is_space(s[q])) q++;
    if (q >= p1 || s[q] != '=') return 0;
    q++;
    while (q < p1 && is_space(s[q])) q++;
    int a = q;
    if (q < p1 && s[q] == '-') q++;
    int d = q;
    while (d < p1 && is_digit(s[d])) d++;
    if (d < p1 && s[d] == '.' && d + 1 < p1 && is_digit(s[d + 1])) {
        d++;
        while (d < p1 && is_digit(s[d])) d++;
    } else if (d == q) {
        return 0;
    }
    *ns = a; *ne = d;
    return 1;
}

static int var_index(const char *s, int a, int b) {
    if (b - a == 10 && !memcmp(s + a, "duty_cycle", 10)) return 0;
    if (b - a == 8 && !memcmp(s + a, "tx_power", 8)) return 1;
    return -2;
}

/* decimal literal -> double. For up to 15 significant digits and 22 decimals
 * M and 10^k are exact doubles, so M / 10^k is the correctly rounded value,
 * identical to Python's float(); avoids pulling strtod (and libc syscalls) */
static double parse_decimal(const char *s, int a, int b) {
    double m = 0, p10 = 1;
    int neg = 0, frac = 0;
    if (a < b && s[a] == '-') { neg = 1; a++; }
    for (; a < b; a++) {
        if (s[a] == '.') { frac = 1; continue; }
        m = m * 10 + (s[a] - '0');
        if (frac) p10 *= 10;
    }
    return neg ? -(m / p10) : m / p10;
}

/* parse_action: -1 none, -2 non-actionable, else actuator index */
static int parse_action(const char *s, int n, double *val) {
    int p;
    for (p = 0; p + 3 <= n; p++) {
        if (s[p] != 'd' || s[p + 1] != 'o' || s[p + 2] != '(') continue;
        int q = p + 3;
        while (q < n && s[q] != '(' && s[q] != ')') q++;
        if (q >= n || s[q] != ')') continue;          /* no match at this start */
        int first = -1, fa = 0, fb = 0, i, vs, ve, ns, ne;
        for (i = p + 3; i < q;) {
            if (do_arg(s, i, q, &vs, &ve, &ns, &ne)) {
                int idx = var_index(s, vs, ve);
                if (first < 0) { first = idx; fa = ns; fb = ne; }
                if (idx >= 0) { fa = ns; fb = ne; first = idx; goto found; }
                i = ne;
            } else {
                i++;
            }
        }
        if (first == -1) { p = q; continue; }         /* no args: next do() match */
    found:
        *val = parse_decimal(s, fa, fb);
        return first;
    }
    return -1;
}

/* EFFECT_PATTERN alternatives, in order; '#' = [_ ] */
static const char *const QV[] = {"end-to-end latency", "energy consumption", "latency", "throughput",
                                 "packet#loss", "energy"};
static const int8_t QV_IDX[] = {0, 3, 0, 1, 2, 3};    /* latency, throughput, packet_loss, energy */
static const char *const DW[] = {"increase", "decrease", "rise", "drop", "reduce", "grow"};
static const int8_t DW_SIGN[] = {1, -1, 1, -1, -1, 1};

static int gap_ok(const char *s, int n, int p) {
    char c = s[p];
    if (c == '\n' || c == ';') return 0;
    if (c == '.') {
        if (p + 1 < n && is_space(s[p + 1])) return 0;
        if (p + 1 == n || (p + 2 == n && s[p + 1] == '\n')) return 0;   /* \.$ */
    }
    return 1;
}

/* parse_claims: fills (qos, sign) pairs; returns the count */
static int parse_claims(const char *s, int n, int8_t qv[], int8_t sg[], int max) {
    int p = 0, m = 0;
    while (p < n) {
        int end = -1, a;
        for (a = 0; a < 6 && end < 0; a++) {
            int L = lit_ci(s, n, p, QV[a]);
            if (!L) continue;
            int g, q = p + L;
            for (g = 0; g <= 80 && end < 0; g++) {
                int pos = q + g;
                if (g > 0 && (pos - 1 >= n || !gap_ok(s, n, pos - 1))) break;
                if (pos > n) break;
                if (pos > 0 && pos < n && is_word(s[pos - 1]) && is_word(s[pos])) continue;
                if (pos >= n || !is_word(s[pos])) continue;  /* \b needs a word char next */
                int w;
                for (w = 0; w < 6; w++) {
                    int K = lit_ci(s, n, pos, DW[w]);
                    if (K && (pos + K >= n || !is_word(s[pos + K]))) {
                        if (m < max) { qv[m] = QV_IDX[a]; sg[m] = DW_SIGN[w]; m++; }
                        end = pos + K;
                        break;
                    }
                }
            }
        }
        p = end > 0 ? end : p + 1;
    }
    return m;
}

void gate_gstar_claims(const char *text, const double cur[2], const int8_t implied[2][4],
                       double nxt[2]) {
    int n = (int)strlen(text), k, nc;
    double value;
    int8_t qv[16], sg[16];
    nxt[0] = cur[0]; nxt[1] = cur[1];
    int a = parse_action(text, n, &value);
    if (a < 0) return;                                  /* UNPARSEABLE / NON_ACTIONABLE */
    if (!(value >= 0.0 && value <= 1.0)) return;        /* OUT_OF_RANGE */
    double cv = cur[a];
    if (fabs(value - cv) > MAX_STEP) value = cv + sgn(value - cv) * MAX_STEP;
    nc = parse_claims(text, n, qv, sg, 16);
    if (nc && fabs(clipd(value, LO, HI) - cv) < NO_OP_EPS) return;   /* NO_OP_CLAIM */
    double mv = sgn(value - cv);
    if (mv == 0) mv = 1;
    for (k = 0; k < nc; k++) {
        int im = implied[a][qv[k]];
        if (im == 0 || im * mv != sg[k]) return;        /* CAUSAL_CHECK_FAILED */
    }
    nxt[a] = clipd(value, LO, HI);
}

/* ------------------------------------------- local-adjusted observational */
#define OBS_P 6
#define OBS_BW 0.12
#define OBS_MIN_N 30

void obs_init(obs_gate_t *g, const double (*hist)[6], int32_t n_hist, double (*live)[6],
              int32_t window, double ci_z) {
    g->hist = hist; g->n_hist = n_hist; g->live = live; g->window = window;
    g->n_live = 0; g->head = 0; g->ci_z = ci_z;
}

void obs_observe(obs_gate_t *g, const double applied[2], const double exo[3], double cost) {
    double *r = g->live[(g->head + g->n_live) % g->window];
    if (g->n_live == g->window) { r = g->live[g->head]; g->head = (g->head + 1) % g->window; }
    else g->n_live++;
    r[0] = applied[0]; r[1] = applied[1]; r[2] = cost; r[3] = exo[0]; r[4] = exo[1]; r[5] = exo[2];
}

typedef struct { double A[OBS_P][OBS_P], b[OBS_P], yy, sw, sw2; } obs_acc_t;

static void acc_row(obs_acc_t *s, const double *r, const double cur[2]) {
    double x[OBS_P] = {r[0], r[1], r[3], r[4], r[5], 1.0};
    double d0 = r[0] - cur[0], d1 = r[1] - cur[1];
    double w = exp(-(d0 * d0 + d1 * d1) / (2 * OBS_BW * OBS_BW)) + 1e-6, y = r[2];
    int i, j;
    for (i = 0; i < OBS_P; i++) {
        double wx = w * x[i];
        for (j = 0; j <= i; j++) s->A[i][j] += wx * x[j];
        s->b[i] += wx * y;
    }
    s->yy += w * y * y; s->sw += w; s->sw2 += w * w;
}

/* solve M z = rhs by Gaussian elimination with partial pivoting */
static void solve(double M[OBS_P][OBS_P], double rhs[OBS_P], double z[OBS_P]) {
    int i, j, k;
    for (k = 0; k < OBS_P; k++) {
        int piv = k;
        for (i = k + 1; i < OBS_P; i++) if (fabs(M[i][k]) > fabs(M[piv][k])) piv = i;
        if (piv != k) {
            for (j = 0; j < OBS_P; j++) { double t = M[k][j]; M[k][j] = M[piv][j]; M[piv][j] = t; }
            double t = rhs[k]; rhs[k] = rhs[piv]; rhs[piv] = t;
        }
        if (fabs(M[k][k]) < 1e-300) continue;
        for (i = k + 1; i < OBS_P; i++) {
            double f = M[i][k] / M[k][k];
            for (j = k; j < OBS_P; j++) M[i][j] -= f * M[k][j];
            rhs[i] -= f * rhs[k];
        }
    }
    for (i = OBS_P - 1; i >= 0; i--) {
        double v = rhs[i];
        for (j = i + 1; j < OBS_P; j++) v -= M[i][j] * z[j];
        z[i] = fabs(M[i][i]) < 1e-300 ? 0.0 : v / M[i][i];
    }
}

/* local effect of actuator j on the cost; returns 0 if no data */
static int obs_effect(obs_gate_t *g, int j, const double cur[2], double *est, double *se) {
    static obs_acc_t s;
    double M[OBS_P][OBS_P], rhs[OBS_P], coef[OBS_P], e[OBS_P], col[OBS_P];
    int i, k, n = g->n_hist;
    memset(&s, 0, sizeof s);
    for (i = 0; i < g->n_hist; i++) acc_row(&s, g->hist[i], cur);
    if (g->n_live >= OBS_MIN_N) {
        for (i = 0; i < g->n_live; i++) acc_row(&s, g->live[(g->head + i) % g->window], cur);
        n += g->n_live;
    }
    if (n == 0) return 0;
    for (i = 0; i < OBS_P; i++) for (k = 0; k < i; k++) s.A[k][i] = s.A[i][k];
    memcpy(M, s.A, sizeof M); memcpy(rhs, s.b, sizeof rhs);
    solve(M, rhs, coef);
    *est = coef[j];
    if (g->ci_z <= 0) return 1;
    /* weighted residual sum of squares, Kish effective sample size */
    double rss = s.yy;
    for (i = 0; i < OBS_P; i++) {
        rss -= 2 * coef[i] * s.b[i];
        for (k = 0; k < OBS_P; k++) rss += coef[i] * s.A[i][k] * coef[k];
    }
    double n_eff = s.sw * s.sw / s.sw2, dof = n_eff - OBS_P;
    for (i = 0; i < OBS_P; i++) e[i] = (i == j);
    memcpy(M, s.A, sizeof M);
    solve(M, e, col);
    double v = rss / (dof > 1.0 ? dof : 1.0) * col[j];
    *se = sqrt(v > 0 ? v : 0.0);
    return 1;
}

void obs_decide(obs_gate_t *g, int pa, double pv, const double cur[2], double nxt[2]) {
    double est, se = 0;
    nxt[0] = cur[0]; nxt[1] = cur[1];
    if (pa < 0 || !(pv >= 0.0 && pv <= 1.0)) return;          /* REJECT */
    double step = clipd(pv - cur[pa], -MAX_STEP, MAX_STEP);
    if (fabs(step) < 1e-9) return;                              /* HOLD */
    if (!obs_effect(g, pa, cur, &est, &se)) return;
    double d = sgn(step);
    if (g->ci_z > 0) {
        /* execute only if the one-sided bound certifies improvement */
        if (d * est + g->ci_z * se < 0) nxt[pa] = clipd(cur[pa] + step, LO, HI);
        return;
    }
    if (d * est < 0) nxt[pa] = clipd(cur[pa] + step, LO, HI);
}
