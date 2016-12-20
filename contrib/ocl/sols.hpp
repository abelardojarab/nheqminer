#pragma once

namespace ocl {
	typedef uint8_t		uchar;
	typedef uint32_t	uint;
	typedef uint64_t	ulong;


	template<int MAXSOLS, int PARAMK>
	struct	sols_s
	{
		uint	nr;
		uint	likely_invalids;
		uchar	valid[MAXSOLS];
		uint	values[MAXSOLS][(1 << PARAMK)];
	};
}

typedef ocl::sols_s<10, 9> sols_t;

