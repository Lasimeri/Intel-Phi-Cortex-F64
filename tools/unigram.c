/*
 * unigram: metric baseline only (never a generator). usage: unigram TRAIN.u16 NTRAIN VALID.u16 NVALID
 * Token counts on the first NTRAIN train tokens (add-one smoothing over 4096); prints the cross-entropy of the first
 * NVALID valid tokens under that distribution in bits/token, and the train unigram entropy.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
	if (argc < 5) { fprintf(stderr, "usage: unigram TRAIN.u16 NTRAIN VALID.u16 NVALID\n"); return 2; }
	const long nt = atol(argv[2]), nv = atol(argv[4]);
	double cnt[4096] = { 0 };
	FILE *f = fopen(argv[1], "rb");
	if (!f) { perror(argv[1]); return 1; }
	uint16_t *buf = malloc((size_t)nt * 2);
	const long got = (long)fread(buf, 2, (size_t)nt, f);
	fclose(f);
	for (long i = 0; i < got; i++) cnt[buf[i]]++;
	double H = 0;
	for (int v = 0; v < 4096; v++) { cnt[v] += 1; }
	for (int v = 0; v < 4096; v++) { const double p = cnt[v] / (got + 4096); H -= p * log2(p); }
	f = fopen(argv[3], "rb");
	if (!f) { perror(argv[3]); return 1; }
	uint16_t *vb = malloc((size_t)nv * 2);
	const long gv = (long)fread(vb, 2, (size_t)nv, f);
	fclose(f);
	double ce = 0;
	for (long i = 0; i < gv; i++) ce -= log2(cnt[vb[i]] / (got + 4096));
	printf("unigram: %ld train tokens, entropy %.4f bits/token; valid %ld tokens cross-entropy %.4f bits/token (%.4f bits/byte at 3.961 B/token)\n",
	       got, H, gv, ce / gv, ce / gv / 3.961);
	return 0;
}
