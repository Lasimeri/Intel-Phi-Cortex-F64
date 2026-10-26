/*
 * coherence: the coach's text metrics. No model output is ever used as training data; this tool only measures.
 *   coherence build LIST.txt MIN TEXT...   word list = lower-cased alphabetic words occurring >= MIN times in the TEXT files
 *   coherence score LIST.txt TEXT          metrics of one text (a generated sample or a reference passage):
 *       words         alphabetic words in the text
 *       hit           share of them present in the list (vocabulary coherence; the list comes from the human corpus)
 *       hit5          the same over words of 5 or more letters (function words no longer carry the score)
 *       big           share of adjacent word pairs that occur in the corpus (the list stores pairs as "a_b", count >= 2)
 *       rep4          share of word 4-grams that already occurred earlier in the text (looping)
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

#define HBITS 22
#define HSIZE (1u << HBITS)
#define WMAX 48
#define SCORE_FMT "hit %.3f  hit5 %.3f  big %.3f  rep4 %.3f  sent %.1f (%.2f in 3..40)  wlen %.1f  words %ld\n"

typedef struct { char w[WMAX]; uint32_t n; } entry;
static entry *tab;

static uint32_t hash(const char *s)
{
	uint32_t h = 2166136261u;
	for (; *s; s++) h = (h ^ (uint8_t)*s) * 16777619u;
	return h;
}

static entry *find(const char *w, int insert)
{
	uint32_t i = hash(w) & (HSIZE - 1);
	for (;;) {
		if (!tab[i].w[0]) { if (!insert) return NULL; strncpy(tab[i].w, w, WMAX - 1); return &tab[i]; }
		if (!strcmp(tab[i].w, w)) return &tab[i];
		i = (i + 1) & (HSIZE - 1);
	}
}

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
	if (strcmp(argv[1], "score")) { fprintf(stderr, "coherence: unknown mode %s\n", argv[1]); return 2; }
	FILE *l = fopen(argv[2], "r");
	if (!l) { perror(argv[2]); return 1; }
	unsigned n;
	while (fscanf(l, "%47s %u", w, &n) == 2) { entry *e = find(w, 1); if (e) e->n = n; }
	fclose(l);
	FILE *f = fopen(argv[3], "rb");
	if (!f) { perror(argv[3]); return 1; }
	/* 4-gram repetition over word hashes */
	enum { G4MAX = 1 << 16 };
	uint64_t *seen = calloc(G4MAX, sizeof *seen);
	uint32_t hist[4] = { 0 }, nh = 0;
	long words = 0, hits = 0, letters = 0, g4 = 0, g4rep = 0, sents = 0, sents_ok = 0, cur = 0, w5 = 0, hit5 = 0, pairs = 0, pairhit = 0;
	char prev[WMAX] = "", pair[WMAX];
	while (next_word(f, w, &se, &wb)) {
		words++;
		letters += (long)strlen(w);
		if (find(w, 0)) hits++;
		if (strlen(w) >= 5) { w5++; if (find(w, 0)) hit5++; }
		if (prev[0] && strlen(prev) + strlen(w) + 2 < WMAX) { snprintf(pair, sizeof pair, "%s_%s", prev, w); pairs++; if (find(pair, 0)) pairhit++; }
		strcpy(prev, se ? "" : w);
		hist[nh++ & 3] = hash(w);
		if (nh >= 4) {
			uint64_t key = ((uint64_t)hist[0] * 1000003u) ^ ((uint64_t)hist[1] << 7) ^ ((uint64_t)hist[2] << 17) ^ ((uint64_t)hist[3] << 29);
			if (!key) key = 1;
			uint32_t i = (uint32_t)(key % G4MAX), found = 0;
			for (;;) { if (!seen[i]) { seen[i] = key; break; } if (seen[i] == key) { found = 1; break; } i = (i + 1) % G4MAX; }
			g4++;
			g4rep += found;
		}
		cur++;
		if (se) { sents++; if (cur >= 3 && cur <= 40) sents_ok++; cur = 0; }
	}
	if (cur) { sents++; if (cur >= 3 && cur <= 40) sents_ok++; }
	fclose(f);
	printf("hit %.3f  hit5 %.3f  big %.3f  rep4 %.3f  sent %.1f (%.2f in 3..40)  wlen %.1f  words %ld\n", words ? (double)hits / words : 0, w5 ? (double)hit5 / w5 : 0, pairs ? (double)pairhit / pairs : 0, g4 ? (double)g4rep / g4 : 0,
	       sents ? (double)words / sents : 0, sents ? (double)sents_ok / sents : 0, words ? (double)letters / words : 0, words);
	return 0;
}
