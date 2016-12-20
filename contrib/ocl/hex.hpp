#pragma once

#include <cstdio>

namespace ocl {
	
inline void hexdump(uint8_t *a, uint32_t a_len)
{
	for (uint32_t i = 0; i < a_len; i++)
		fprintf(stderr, "%02x", a[i]);
}

inline char *s_hexdump(const void *_a, uint32_t a_len)
{
	const uint8_t	*a = (uint8_t	*)_a;
	static char		buf[1024];
	uint32_t		i;
	for (i = 0; i < a_len && i + 2 < sizeof(buf); i++)
		sprintf(buf + i * 2, "%02x", a[i]);
	buf[i * 2] = 0;
	return buf;
}

inline uint8_t hex2val(const char *base, size_t off)
{
	const char          c = base[off];
	if (c >= '0' && c <= '9')           return c - '0';
	else if (c >= 'a' && c <= 'f')      return 10 + c - 'a';
	else if (c >= 'A' && c <= 'F')      return 10 + c - 'A';
	printf("Invalid hex char at offset %zd: ...%c...\n", off, c);
	return 0;
}

}
