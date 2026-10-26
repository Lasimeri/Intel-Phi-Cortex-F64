/*
 * coherence.h: the coach's text metrics as a header (used by tools/coherence.c and host/coach.c). No model output is ever
 * training data; this only measures. A word is a maximal run of ASCII letters and apostrophes, lower-cased.
 *   coh_load(LIST)            loads a word + bigram list written by `coherence build` (pairs stored as "a_b")
 *   coh_score(text, n, &m)    metrics of one text: hit, hit5, big, rep4 (inside 256-word windows), sent, sent_ok, wlen, words
 */
#ifndef COH_H
#define COH_H
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COH_HBITS 23
#define COH_HSIZE (1u << COH_HBITS)
#define COH_WMAX 48
#define COH_G4MAX (1 << 12)
#define COH_G4WIN 256

typedef struct { char w[COH_WMAX]; uint32_t n; } coh_entry;
static coh_entry *coh_tab;

static inline uint32_t coh_hash(const char *s) { uint32_t h = 2166136261u; for (; *s; s++) h = (h ^ (uint8_t)*s) * 16777619u; return h; }

static inline coh_entry *coh_find(const char *w, int insert)
{
	if (!coh_tab) { coh_tab = calloc(COH_HSIZE, sizeof *coh_tab); if (!coh_tab) return NULL; }
	uint32_t i = coh_hash(w) & (COH_HSIZE - 1);
	for (uint32_t probes = 0; probes < COH_HSIZE; probes++) {
		if (!coh_tab[i].w[0]) { if (!insert) return NULL; strncpy(coh_tab[i].w, w, COH_WMAX - 1); return &coh_tab[i]; }
		if (!strcmp(coh_tab[i].w, w)) return &coh_tab[i];
		i = (i + 1) & (COH_HSIZE - 1);
	}
	return NULL;
}

static inline int coh_load(const char *path)
{
	FILE *l = fopen(path, "r");
	if (!l) return -1;
	char w[COH_WMAX]; unsigned n; long k = 0;
	while (fscanf(l, "%47s %u", w, &n) == 2) { coh_entry *e = coh_find(w, 1); if (e) { e->n = n; k++; } }
	fclose(l);
	return k > 0 ? 0 : -1;
}

/* next word from a buffer; *se set when the word is followed by . ! ? */
static inline int coh_next(const char **p, const char *end, char *w, int *se)
{
	int n = 0; *se = 0;
	while (*p < end) {
		const int c = (unsigned char)**p; (*p)++;
		if (isalpha(c) || (c == '\'' && n)) { if (n < COH_WMAX - 1) w[n++] = (char)tolower(c); }
		else if (n) { if (c == '.' || c == '!' || c == '?') *se = 1; break; }
	}
	w[n] = 0;
	return n;
}

typedef struct { double hit, hit5, big, rep4, sent, sent_ok, wlen; long words; } coh_metrics;

static inline void coh_score(const char *text, size_t len, coh_metrics *m)
{
	const char *p = text, *end = text + len;
	char w[COH_WMAX], prev[COH_WMAX] = "", pair[COH_WMAX * 2];
	int se;
	static uint64_t seen[COH_G4MAX];
	uint32_t hist[4] = { 0 }, nh = 0;
	long words = 0, hits = 0, letters = 0, g4 = 0, g4rep = 0, sents = 0, sents_ok = 0, cur = 0, w5 = 0, hit5 = 0, pairs = 0, pairhit = 0, win = 0;
	memset(seen, 0, sizeof seen);
	while (coh_next(&p, end, w, &se)) {
		if (win++ == COH_G4WIN) { memset(seen, 0, sizeof seen); win = 1; nh = 0; }
		words++;
		letters += (long)strlen(w);
		const int in = coh_find(w, 0) != NULL;
		hits += in;
		if (strlen(w) >= 5) { w5++; hit5 += in; }
		if (prev[0] && strlen(prev) + strlen(w) + 2 < COH_WMAX) { snprintf(pair, sizeof pair, "%s_%s", prev, w); pairs++; if (coh_find(pair, 0)) pairhit++; }
		strcpy(prev, se ? "" : w);
		hist[nh++ & 3] = coh_hash(w);
		if (nh >= 4) {
			uint64_t key = ((uint64_t)hist[0] * 1000003u) ^ ((uint64_t)hist[1] << 7) ^ ((uint64_t)hist[2] << 17) ^ ((uint64_t)hist[3] << 29);
			if (!key) key = 1;
			uint32_t i = (uint32_t)(key % COH_G4MAX), found = 0;
			for (long probes = 0; probes < COH_G4MAX; probes++, i = (i + 1) % COH_G4MAX) { if (!seen[i]) { seen[i] = key; break; } if (seen[i] == key) { found = 1; break; } }
			g4++; g4rep += found;
		}
		cur++;
		if (se) { sents++; if (cur >= 3 && cur <= 40) sents_ok++; cur = 0; }
	}
	if (cur) { sents++; if (cur >= 3 && cur <= 40) sents_ok++; }
	m->words = words;
	m->hit = words ? (double)hits / words : 0;
	m->hit5 = w5 ? (double)hit5 / w5 : 0;
	m->big = pairs ? (double)pairhit / pairs : 0;
	m->rep4 = g4 ? (double)g4rep / g4 : 0;
	m->sent = sents ? (double)words / sents : 0;
	m->sent_ok = sents ? (double)sents_ok / sents : 0;
	m->wlen = words ? (double)letters / words : 0;
}

static inline void coh_print(FILE *f, const coh_metrics *m)
{
	fprintf(f, "hit %.3f  hit5 %.3f  big %.3f  rep4 %.3f  sent %.1f (%.2f in 3..40)  wlen %.1f  words %ld\n", m->hit, m->hit5, m->big, m->rep4, m->sent, m->sent_ok, m->wlen, m->words);
}
#endif
