/*
 * jsonl-text: dataset conversion (host, C). usage: jsonl-text IN.jsonl OUT.txt [MAX_NONASCII_PERMILLE]
 * Each input line is a JSON object with a "text" field (the guanaco form of OpenAssistant's oasst1: a conversation as
 * "### Human: ...### Assistant: ..."). The field is unescaped (\n \t \" \\ \/ \uXXXX to UTF-8) and written as one
 * document followed by "<|endoftext|>\n" (the tokenizer's end marker). A document whose share of non-ASCII bytes is
 * above MAX_NONASCII_PERMILLE (default 20 = 2 %) is skipped: a crude English filter for a 35-language corpus.
 * Prints documents kept and skipped, bytes written.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int hexv(int c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; }

static int put_utf8(unsigned char *o, unsigned cp)
{
	if (cp < 0x80) { o[0] = (unsigned char)cp; return 1; }
	if (cp < 0x800) { o[0] = 0xC0 | cp >> 6; o[1] = 0x80 | (cp & 0x3F); return 2; }
	if (cp < 0x10000) { o[0] = 0xE0 | cp >> 12; o[1] = 0x80 | (cp >> 6 & 0x3F); o[2] = 0x80 | (cp & 0x3F); return 3; }
	o[0] = 0xF0 | cp >> 18; o[1] = 0x80 | (cp >> 12 & 0x3F); o[2] = 0x80 | (cp >> 6 & 0x3F); o[3] = 0x80 | (cp & 0x3F);
	return 4;
}

/* the unescaped value of the "text" field of one JSON line into out; its length, or -1 */
static long text_field(const char *line, unsigned char *out, long cap)
{
	const char *p = strstr(line, "\"text\"");
	if (!p) return -1;
	p = strchr(p + 6, ':');
	if (!p) return -1;
	p++;
	while (*p == ' ') p++;
	if (*p != '"') return -1;
	p++;
	long n = 0;
	while (*p && *p != '"' && n + 4 < cap) {
		if (*p != '\\') { out[n++] = (unsigned char)*p++; continue; }
		p++;
		switch (*p) {
		case 'n': out[n++] = '\n'; p++; break;
		case 't': out[n++] = '\t'; p++; break;
		case 'r': out[n++] = '\r'; p++; break;
		case 'b': out[n++] = '\b'; p++; break;
		case 'f': out[n++] = '\f'; p++; break;
		case '"': out[n++] = '"'; p++; break;
		case '\\': out[n++] = '\\'; p++; break;
		case '/': out[n++] = '/'; p++; break;
		case 'u': {
			unsigned cp = 0;
			for (int i = 1; i <= 4; i++) { const int h = hexv(p[i]); if (h < 0) return -1; cp = cp << 4 | (unsigned)h; }
			p += 5;
			if (cp >= 0xD800 && cp < 0xDC00 && p[0] == '\\' && p[1] == 'u') {   /* surrogate pair */
				unsigned lo = 0;
				for (int i = 2; i <= 5; i++) { const int h = hexv(p[i]); if (h < 0) return -1; lo = lo << 4 | (unsigned)h; }
				p += 6;
				cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
			}
			n += put_utf8(out + n, cp);
			break;
		}
		default: return -1;
		}
	}
	return *p == '"' ? n : -1;
}

int main(int argc, char **argv)
{
	if (argc < 3) { fprintf(stderr, "usage: jsonl-text IN.jsonl OUT.txt [MAX_NONASCII_PERMILLE]\n"); return 2; }
	const int maxpm = argc > 3 ? atoi(argv[3]) : 20;
	FILE *in = fopen(argv[1], "r"), *out = fopen(argv[2], "wb");
	if (!in || !out) { perror(in ? argv[2] : argv[1]); return 1; }
	static char line[1 << 22];
	static unsigned char buf[1 << 22];
	long kept = 0, skipped = 0, bad = 0, bytes = 0;
	while (fgets(line, sizeof line, in)) {
		const long n = text_field(line, buf, sizeof buf);
		if (n < 0) { bad++; continue; }
		long na = 0;
		for (long i = 0; i < n; i++) na += buf[i] >= 0x80;
		if (n == 0 || na * 1000 > (long)maxpm * n) { skipped++; continue; }
		/* turn markers -> Anthropic format */
		for (long i = 0; i < n; ) {
			if (i + 11 <= n && !memcmp(buf + i, "### Human: ", 11)) { fputs("\n\nHuman: ", out); bytes += 9; i += 11; }
			else if (i + 15 <= n && !memcmp(buf + i, "### Assistant: ", 15)) { fputs("\n\nAssistant: ", out); bytes += 13; i += 15; }
			else { fputc(buf[i], out); bytes++; i++; }
		}
		fputs("\n<|endoftext|>\n", out);
		bytes += 15;
		kept++;
	}
	fclose(in);
	fclose(out);
	printf("jsonl-text: %ld documents kept, %ld skipped (non-ASCII share above %d permille), %ld unparsable; %ld bytes -> %s\n", kept, skipped, maxpm, bad, bytes, argv[2]);
	return 0;
}
