/*
 * stats: host side of the timing gates. usage: stats TICK_MHZ FILE...
 * Each FILE holds raw u32 TSC tick counts from the card (1 tick = 1 / TICK_MHZ us; 1238.093 MHz measured in s00b).
 * Prints one line per file: n, min, median, p99, p99.9, max (ns) and the mean.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int cmpu(const void *a, const void *b)
{
	const uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
	return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
	if (argc < 3) { fprintf(stderr, "usage: stats TICK_MHZ FILE...\n"); return 2; }
	const double ns = 1000.0 / atof(argv[1]);
	printf("%-28s %8s %9s %9s %9s %9s %9s %9s\n", "file", "n", "min ns", "median", "p99", "p99.9", "max", "mean");
	for (int a = 2; a < argc; a++) {
		FILE *f = fopen(argv[a], "rb");
		if (!f) { perror(argv[a]); return 1; }
		fseek(f, 0, SEEK_END);
		const long n = ftell(f) / 4;
		fseek(f, 0, SEEK_SET);
		uint32_t *v = malloc((size_t)n * 4);
		if (n <= 0 || fread(v, 4, (size_t)n, f) != (size_t)n) { fprintf(stderr, "stats: read %s\n", argv[a]); return 1; }
		fclose(f);
		double sum = 0;
		for (long i = 0; i < n; i++) sum += v[i];
		qsort(v, (size_t)n, 4, cmpu);
		const char *base = argv[a];
		for (const char *p = argv[a]; *p; p++) if (*p == '/') base = p + 1;
		printf("%-28s %8ld %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f\n", base, n, v[0] * ns, v[n / 2] * ns, v[(long)(n * 0.99)] * ns,
		       v[(long)(n * 0.999)] * ns, v[n - 1] * ns, sum / n * ns);
		free(v);
	}
	return 0;
}
