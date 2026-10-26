/*
 * coherence: the coach's text metrics. No model output is ever used as training data; this tool only measures.
 *   coherence build LIST.txt MIN TEXT...   word list = lower-cased alphabetic words occurring >= MIN times in the TEXT files
 *   coherence score LIST.txt TEXT          metrics of one text (a generated sample or a reference passage):
 *   coherence windows LIST.txt TEXT        the same metrics per 256-word window of a long reference text: mean, sd, p10/p50/p90
 *                                          of hit5, big and rep4 (the human band the coach compares samples against)
 *       words         alphabetic words in the text
 *       hit           share of them present in the list (vocabulary coherence; the list comes from the human corpus)
 *       hit5          the same over words of 5 or more letters (function words no longer carry the score)
 *       big           share of adjacent word pairs that occur in the corpus (the list stores pairs as "a_b", count >= 2)
 *       rep4          share of word 4-grams that already occurred earlier in the same 256-word window (looping)
 *       sent          mean words per sentence (split at . ! ?), and the share of sentences of 3..40 words
 *       wlen          mean word length in letters
 *   Prints one line: "hit 0.812  rep4 0.031  sent 11.2 (0.90 in 3..40)  wlen 4.3  words 213".
 * A word is a maximal run of ASCII letters and apostrophes, lower-cased. Everything is plain C, no dependencies.
 */
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "coherence.h"
#define HBITS COH_HBITS
#define HSIZE COH_HSIZE
#define WMAX COH_WMAX

typedef coh_entry entry;
#define tab coh_tab

#define hash coh_hash

#define find coh_find

/* next word from a stream; returns length (0 at EOF); sets *sentence_end when the word was followed by . ! ? */
static int next_word(FILE *f, char *w, int *sentence_end, int *was_end_before)
{
	int c, n = 0;
	*sentence_end = 0;
	*was_end_before = 0;
	while ((c = fgetc(f)) != EOF) {
		if (isalpha(c) || (c == '\'' && n)) { if (n < WMAX - 1) w[n++] = (char)tolower(c); }
		else if (n) { if (c == '.' || c == '!' || c == '?') *sentence_end = 1; break; }
		else if (c == '.' || c == '!' || c == '?') *was_end_before = 1;
	}
	w[n] = 0;
	return n;
}

static int cmp_entry(const void *a, const void *b) { return strcmp(((const entry *)a)->w, ((const entry *)b)->w); }
static int cmpd(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv)
{
	if (argc < 4) { fprintf(stderr, "usage: coherence build LIST MIN TEXT... | coherence score LIST TEXT\n"); return 2; }
	tab = calloc(HSIZE, sizeof *tab);
	if (!tab) return 1;
	char w[WMAX];
	int se, wb;
	if (!strcmp(argv[1], "build")) {
		const int min = atoi(argv[3]);
		long total = 0;
		for (int a = 4; a < argc; a++) {
			FILE *f = fopen(argv[a], "rb");
			if (!f) { perror(argv[a]); return 1; }
			char prev[WMAX] = "", pair[WMAX];
			while (next_word(f, w, &se, &wb)) {
				entry *e = find(w, 1);
				if (e) { e->n++; total++; }
				if (prev[0] && strlen(prev) + strlen(w) + 2 < WMAX) { snprintf(pair, sizeof pair, "%s_%s", prev, w); entry *p = find(pair, 1); if (p) p->n++; }
				strcpy(prev, se ? "" : w);   /* no pairs across sentence ends */
			}
			fclose(f);
		}
		qsort(tab, HSIZE, sizeof *tab, cmp_entry);   /* empties ("") sort first */
		FILE *o = fopen(argv[2], "w");
		if (!o) { perror(argv[2]); return 1; }
		long kept = 0, distinct = 0;
		for (uint32_t i = 0; i < HSIZE; i++) if (tab[i].w[0]) { const int pair = strchr(tab[i].w, (int)0x5f) != NULL; distinct += !pair; if (tab[i].n >= (uint32_t)(pair ? 2 : min)) { fprintf(o, "%s %u\n", tab[i].w, tab[i].n); kept += !pair; } }
		fclose(o);
		printf("coherence build: %ld word tokens, %ld distinct, %ld kept (count >= %d) -> %s\n", total, distinct, kept, min, argv[2]);
		return 0;
	}
	const int windows = !strcmp(argv[1], "windows");
	if (strcmp(argv[1], "score") && !windows) { fprintf(stderr, "coherence: unknown mode %s\n", argv[1]); return 2; }
	if (coh_load(argv[2])) { perror(argv[2]); return 1; }
	FILE *f = fopen(argv[3], "rb");
	if (!f) { perror(argv[3]); return 1; }
	fseek(f, 0, SEEK_END);
	const long len = ftell(f);
	fseek(f, 0, SEEK_SET);
	char *text = malloc((size_t)len + 1);
	if (!text || (long)fread(text, 1, (size_t)len, f) != len) { perror(argv[3]); return 1; }
	fclose(f);
	coh_metrics m;
	if (!windows) { coh_score(text, (size_t)len, &m); coh_print(stdout, &m); return 0; }
	/* windows: metrics of every 256-word window, then mean, sd and the 10th / 90th percentiles of hit5, big, rep4 */
	const char *p = text, *end = text + len, *ws = text;
	char w2[COH_WMAX]; int se2; long n = 0, cap = 1024;
	double *h5 = malloc(cap * sizeof *h5), *bg = malloc(cap * sizeof *bg), *r4 = malloc(cap * sizeof *r4);
	long cnt = 0;
	while (coh_next(&p, end, w2, &se2)) {
		if (++cnt == COH_G4WIN) {
			coh_score(ws, (size_t)(p - ws), &m);
			if (n == cap) { cap *= 2; h5 = realloc(h5, cap * sizeof *h5); bg = realloc(bg, cap * sizeof *bg); r4 = realloc(r4, cap * sizeof *r4); }
			h5[n] = m.hit5; bg[n] = m.big; r4[n] = m.rep4; n++;
			ws = p; cnt = 0;
		}
	}
	if (!n) { fprintf(stderr, "coherence windows: text shorter than %d words\n", COH_G4WIN); return 1; }
	const char *names[3] = { "hit5", "big", "rep4" }; double *arr[3] = { h5, bg, r4 };
	printf("windows of %d words: %ld\n", COH_G4WIN, n);
	for (int k = 0; k < 3; k++) {
		double s = 0, ss = 0;
		for (long i = 0; i < n; i++) { s += arr[k][i]; ss += arr[k][i] * arr[k][i]; }
		qsort(arr[k], (size_t)n, sizeof(double), cmpd);
		const double mean = s / n, var = ss / n - mean * mean;
		printf("  %-4s mean %.3f  sd %.3f  p10 %.3f  p50 %.3f  p90 %.3f\n", names[k], mean, var > 0 ? __builtin_sqrt(var) : 0, arr[k][n / 10], arr[k][n / 2], arr[k][(long)(n * 0.9)]);
	}
	return 0;
}
