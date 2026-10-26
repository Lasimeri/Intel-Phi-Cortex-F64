/*
 * ckstat: inspects an F64 cluster checkpoint (120 x CX_NPARAM column parameters, cg, ETB in vocabulary-block layout).
 * usage: ckstat CKPT.bin
 * Prints, per parameter block of every column: max |value|, and the count of non-finite values; then the columns with any
 * non-finite value, the cg and ETB statistics, and the vocabulary rows of the embedding with non-finite entries.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../lib/cx-col.h"
#define CX_ETB(v, c) ((((long)(v) >> 4) * CX_DE + (c)) * 16 + ((v) & 15))

typedef struct { const char *name; int off, n; } blk;
static const blk blocks[] = { { "P_in", CX_O_PIN, CX_DE * CX_DP }, { "Wqb", CX_O_WQB, CX_DP * CX_DB }, { "gain", CX_O_GAIN, CX_DX }, { "W", CX_O_W, CX_DX * CX_NW },
	{ "W1", CX_O_W1, CX_DM * CX_DH }, { "b1", CX_O_B1, CX_DH }, { "Wu", CX_O_WU, CX_DH * CX_DU }, { "wg", CX_O_WG, CX_DH }, { "bg", CX_O_BG, 16 }, { "Wb", CX_O_WB, CX_DH * CX_DB } };

int main(int argc, char **argv)
{
	if (argc != 2) { fprintf(stderr, "usage: ckstat CKPT.bin\n"); return 2; }
	const long n = (long)CX_NCOL * CX_NPARAM + CX_DU + (long)CX_VOCAB * CX_DE;
	double *ck = malloc(n * sizeof *ck);
	FILE *f = fopen(argv[1], "rb");
	if (!f || fread(ck, sizeof *ck, (size_t)n, f) != (size_t)n) { perror(argv[1]); return 1; }
	fclose(f);
	const int nb = (int)(sizeof blocks / sizeof *blocks);
	double bmax[16] = { 0 };
	long bbad[16] = { 0 };
	int badcols = 0;
	for (int c = 0; c < CX_NCOL; c++) {
		const double *p = ck + (long)c * CX_NPARAM;
		long bad = 0;
		for (int b = 0; b < nb; b++)
			for (int i = 0; i < blocks[b].n; i++) {
				const double v = p[blocks[b].off + i];
				if (!isfinite(v)) { bbad[b]++; bad++; }
				else if (fabs(v) > bmax[b]) bmax[b] = fabs(v);
			}
		if (bad) { if (badcols < 8) printf("column %3d: %ld non-finite values\n", c, bad); badcols++; }
	}
	printf("columns with non-finite values: %d of %d\n", badcols, CX_NCOL);
	for (int b = 0; b < nb; b++) printf("  %-5s max |v| %.4g   non-finite %ld\n", blocks[b].name, bmax[b], bbad[b]);
	const double *cg = ck + (long)CX_NCOL * CX_NPARAM, *E = cg + CX_DU;
	double m = 0; long bad = 0;
	for (int i = 0; i < CX_DU; i++) { if (!isfinite(cg[i])) bad++; else if (fabs(cg[i]) > m) m = fabs(cg[i]); }
	printf("cg: max |v| %.4g  non-finite %ld\n", m, bad);
	m = 0; bad = 0; long badrows = 0;
	for (int v = 0; v < CX_VOCAB; v++) {
		long rb = 0;
		for (int c = 0; c < CX_DE; c++) { const double x = E[CX_ETB(v, c)]; if (!isfinite(x)) rb++; else if (fabs(x) > m) m = fabs(x); }
		if (rb) { bad += rb; if (badrows < 8) printf("  E row %4d: %ld non-finite\n", v, rb); badrows++; }
	}
	printf("E: max |v| %.4g  non-finite %ld in %ld rows\n", m, bad, badrows);
	return 0;
}
