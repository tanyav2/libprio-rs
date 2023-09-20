#include "./fp_u64.cuh"
#include "../fft/fft.cuh"
#include "../fft/twiddles.cuh"
#include "../fft/bitrev_permutation.cuh"
#include "../utils.h"

namespace p64
{
    using Fp = Fp64;
}

extern "C"
{
    __global__ void radix2_dit_butterfly( p64::Fp *input, 
                                             const p64::Fp *twiddles,
                                             const int stage,
                                             const int butterfly_count)
    {
        _radix2_dit_butterfly<p64::Fp>(input, twiddles, stage, butterfly_count);
    }
    
    __global__ void calc_twiddles(p64::Fp *result, const p64::Fp &_omega, const int count)
    {
        _calc_twiddles<p64::Fp>(result, _omega, count);
    };

    __global__ void calc_twiddles_bitrev(p64::Fp *result,
                                            const p64::Fp &_omega,
                                            const int count)
    {
        _calc_twiddles_bitrev<p64::Fp>(result, _omega, count);
    };

    __global__ void bitrev_permutation(
        const p64::Fp *input,
        p64::Fp *result,
        const int len
    ) {
        _bitrev_permutation<p64::Fp>(input, result, len);
    };
}
