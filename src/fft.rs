// SPDX-License-Identifier: MPL-2.0

//! This module implements an iterative FFT algorithm for computing the (inverse) Discrete Fourier
//! Transform (DFT) over a slice of field elements.

use crate::field::FftFriendlyFieldElement;
use crate::fp::{log2, MAX_ROOTS};

// use crate::lambdaworks_math::fft::gpu::cuda::{evaluate_fft_cuda, interpolate_fft_cuda};
use lambdaworks_math::fft::polynomial::{interpolate_fft_cpu, FFTPoly};
use lambdaworks_math::field::element;
use lambdaworks_math::field::test_fields::u64_test_field::U64TestField;
use lambdaworks_math::polynomial::Polynomial;
use std::convert::TryFrom;

/// An error returned by an FFT operation.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum FftError {
    /// The output is too small.
    #[error("output slice is smaller than specified size")]
    OutputTooSmall,
    /// The specified size is too large.
    #[error("size is larger than than maximum permitted")]
    SizeTooLarge,
    /// The specified size is not a power of 2.
    #[error("size is not a power of 2")]
    SizeInvalid,
}

/// Sets `outp` to the DFT of `inp`.
///
/// Interpreting the input as the coefficients of a polynomial, the output is equal to the input
/// evaluated at points `p^0, p^1, ... p^(size-1)`, where `p` is the `2^size`-th principal root of
/// unity.
#[allow(clippy::many_single_char_names)]
pub fn discrete_fourier_transform<F: FftFriendlyFieldElement>(
    outp: &mut [F],
    inp: &[F],
    size: usize,
) -> Result<(), FftError> {
    let d = usize::try_from(log2(size as u128)).map_err(|_| FftError::SizeTooLarge)?;

    if size > outp.len() {
        return Err(FftError::OutputTooSmall);
    }

    if size > 1 << MAX_ROOTS {
        return Err(FftError::SizeTooLarge);
    }

    if size != 1 << d {
        return Err(FftError::SizeInvalid);
    }

    #[allow(clippy::needless_range_loop)]
    for i in 0..size {
        let j = bitrev(d, i);
        outp[i] = if j < inp.len() { inp[j] } else { F::zero() }
    }

    let mut w: F;
    for l in 1..d + 1 {
        w = F::one();
        let r = F::root(l).unwrap();
        let y = 1 << (l - 1);
        let chunk = (size / y) >> 1;

        // unrolling first iteration of i-loop.
        for j in 0..chunk {
            let x = j << l;
            let u = outp[x];
            let v = outp[x + y];
            outp[x] = u + v;
            outp[x + y] = u - v;
        }

        for i in 1..y {
            w *= r;
            for j in 0..chunk {
                let x = (j << l) + i;
                let u = outp[x];
                let v = w * outp[x + y];
                outp[x] = u + v;
                outp[x + y] = u - v;
            }
        }
    }

    Ok(())
}

fn prio_to_lambda_field<F: FftFriendlyFieldElement>(
    input: &F,
) -> element::FieldElement<U64TestField> {
    let temp = input.get_encoded();
    let temp_slice: [u8; 8] = temp.as_slice().try_into().unwrap();
    let temp_u64 = u64::from_be_bytes(temp_slice);
    let element = element::FieldElement::<U64TestField>::from(temp_u64);
    element
}

fn lambda_to_prio_field<F: FftFriendlyFieldElement>(
    input: &element::FieldElement<U64TestField>,
) -> F {
    let temp = input.value();
    let temp_be_bytes = temp.to_be_bytes();
    let prio_field_elem = F::try_from(&temp_be_bytes).unwrap();
    prio_field_elem
}

/// Sets `outp` to the DFT of `inp`.
///
/// Interpreting the input as the coefficients of a polynomial, the output is equal to the input
/// evaluated at points `p^0, p^1, ... p^(size-1)`, where `p` is the `2^size`-th principal root of
/// unity.
#[allow(clippy::many_single_char_names)]
pub fn discrete_fourier_transform_cuda<F: FftFriendlyFieldElement>(
    outp: &mut [F],
    inp: &[F],
    size: usize,
) -> Result<(), FftError> {
    let lambda_field_elements: Vec<element::FieldElement<U64TestField>> = inp
        .to_vec()
        .iter()
        .map(|x| prio_to_lambda_field(x))
        .collect();
    let poly = Polynomial::new(&lambda_field_elements);
    let blowup_factor = size / lambda_field_elements.len();
    let lambda_out = poly.evaluate_fft(blowup_factor, None).unwrap();
    let out: Vec<F> = lambda_out.iter().map(|x| lambda_to_prio_field(x)).collect();
    outp.copy_from_slice(&out);

    Ok(())
}

/// Sets `outp` to the inverse of the DFT of `inp`.
#[cfg(test)]
pub(crate) fn discrete_fourier_transform_inv<F: FftFriendlyFieldElement>(
    outp: &mut [F],
    inp: &[F],
    size: usize,
) -> Result<(), FftError> {
    let size_inv = F::from(F::Integer::try_from(size).unwrap()).inv();
    discrete_fourier_transform(outp, inp, size)?;
    discrete_fourier_transform_inv_finish(outp, size, size_inv);
    Ok(())
}

/// Sets `outp` to the inverse of the DFT of `inp`.
// #[cfg(test)]
pub(crate) fn discrete_fourier_transform_inv_cuda<F: FftFriendlyFieldElement>(
    outp: &mut [F],
    inp: &[F],
    size: usize,
) -> Result<(), FftError> {
    let lambda_field_elements: Vec<element::FieldElement<U64TestField>> = inp
        .to_vec()
        .into_iter()
        .map(|x| prio_to_lambda_field(&x))
        .collect();
    let lambda_out = interpolate_fft_cpu(&lambda_field_elements).unwrap();
    let coeffs = lambda_out.coefficients();
    let out: Vec<F> = coeffs
        .into_iter()
        .map(|x| lambda_to_prio_field(x))
        .collect();
    outp.clone_from_slice(&out);
    Ok(())
}

/// An intermediate step in the computation of the inverse DFT. Exposing this function allows us to
/// amortize the cost the modular inverse across multiple inverse DFT operations.
pub(crate) fn discrete_fourier_transform_inv_finish<F: FftFriendlyFieldElement>(
    outp: &mut [F],
    size: usize,
    size_inv: F,
) {
    let mut tmp: F;
    outp[0] *= size_inv;
    outp[size >> 1] *= size_inv;
    for i in 1..size >> 1 {
        tmp = outp[i] * size_inv;
        outp[i] = outp[size - i] * size_inv;
        outp[size - i] = tmp;
    }
}

// An intermediate step in the computation of the inverse DFT. Exposing this function allows us to
// amortize the cost the modular inverse across multiple inverse DFT operations.
// pub(crate) fn discrete_fourier_transform_inv_finish_cuda<F: FftFriendlyFieldElement>(
//     outp: &mut [F],
//     size: usize,
//     size_inv: F,
// ) {
//     let lambda_size_inv = [prio_to_lambda_field(&size_inv)];
//     let lambda_out = interpolate_fft_cpu(&lambda_size_inv).unwrap();
//     let coeffs = lambda_out.coefficients();
//     let out: Vec<F> = coeffs
//         .into_iter()
//         .map(|x| lambda_to_prio_field(x))
//         .collect();
//     outp.clone_from_slice(&out);
// }

// bitrev returns the first d bits of x in reverse order. (Thanks, OEIS! https://oeis.org/A030109)
fn bitrev(d: usize, x: usize) -> usize {
    let mut y = 0;
    for i in 0..d {
        y += ((x >> i) & 1) << (d - i);
    }
    y >> 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{random_vector, split_vector, Field128, Field64, FieldElement, FieldPrio2};
    use crate::polynomial::{poly_fft, PolyAuxMemory};

    fn discrete_fourier_transform_then_inv_test<F: FftFriendlyFieldElement>() -> Result<(), FftError>
    {
        let test_sizes = [1, 2, 4, 8, 16, 256, 1024, 2048];

        for size in test_sizes.iter() {
            let mut tmp = vec![F::zero(); *size];
            let mut got = vec![F::zero(); *size];
            let want = random_vector(*size).unwrap();

            discrete_fourier_transform(&mut tmp, &want, want.len())?;
            discrete_fourier_transform_inv(&mut got, &tmp, tmp.len())?;
            assert_eq!(got, want);
        }

        Ok(())
    }

    fn discrete_fourier_transform_then_inv_cuda_test<F: FftFriendlyFieldElement>(
    ) -> Result<(), FftError> {
        let test_sizes = [1, 2, 4, 8, 16, 256, 1024, 2048];

        for size in test_sizes.iter() {
            let mut tmp = vec![F::zero(); *size];
            // let mut tmp_plain = vec![F::zero(); (*size) * 2];
            let mut got = vec![F::zero(); *size];
            // let mut got_plain = vec![F::zero(); *size];
            let want = random_vector(*size).unwrap();

            discrete_fourier_transform_cuda(&mut tmp, &want, want.len())?;
            // discrete_fourier_transform(&mut tmp_plain, &want, want.len() * 2)?;
            // assert_eq!(vec![F::zero(); *size], tmp_plain);

            // run both with non cuda input
            discrete_fourier_transform_inv_cuda(&mut got, &tmp, tmp.len())?;
            // discrete_fourier_transform_inv(&mut got_plain, &tmp_plain, tmp.len())?;
            assert_eq!(got, want);
            // assert_eq!(tmp, tmp_plain);
            // assert_eq!(got, got_plain);
        }

        Ok(())
    }

    #[test]
    fn test_field64_cuda() {
        discrete_fourier_transform_then_inv_cuda_test::<Field64>().expect("unexpected error");
    }

    fn lambdaworks_field_conv<F: FftFriendlyFieldElement>() {
        let input: Vec<F> = random_vector(3).unwrap();
        let lambda_field_elements: Vec<element::FieldElement<U64TestField>> = input
            .clone()
            .into_iter()
            .map(|x| prio_to_lambda_field(&x))
            .collect();
        let out: Vec<F> = lambda_field_elements
            .into_iter()
            .map(|x| lambda_to_prio_field(&x))
            .collect();
        assert_eq!(input, out);
    }

    #[test]
    fn test_field64_lambda_cuda() {
        lambdaworks_field_conv::<Field64>();
    }

    #[test]
    fn test_priov2_field32() {
        discrete_fourier_transform_then_inv_test::<FieldPrio2>().expect("unexpected error");
    }

    #[test]
    fn test_field64() {
        discrete_fourier_transform_then_inv_test::<Field64>().expect("unexpected error");
    }

    #[test]
    fn test_field128() {
        discrete_fourier_transform_then_inv_test::<Field128>().expect("unexpected error");
    }

    #[test]
    fn test_priov2_field32_cuda() {
        discrete_fourier_transform_then_inv_cuda_test::<FieldPrio2>().expect("unexpected error");
    }

    #[test]
    fn test_field128_cuda() {
        discrete_fourier_transform_then_inv_cuda_test::<Field128>().expect("unexpected error");
    }

    #[test]
    fn test_recursive_fft() {
        let size = 128;
        let mut mem = PolyAuxMemory::new(size / 2);

        let inp = random_vector(size).unwrap();
        let mut want = vec![FieldPrio2::zero(); size];
        let mut got = vec![FieldPrio2::zero(); size];

        discrete_fourier_transform::<FieldPrio2>(&mut want, &inp, inp.len()).unwrap();

        poly_fft(
            &mut got,
            &inp,
            &mem.roots_2n,
            size,
            false,
            &mut mem.fft_memory,
        );

        assert_eq!(got, want);
    }

    // This test demonstrates a consequence of \[BBG+19, Fact 4.4\]: interpolating a polynomial
    // over secret shares and summing up the coefficients is equivalent to interpolating a
    // polynomial over the plaintext data.
    #[test]
    fn test_fft_linearity() {
        let len = 16;
        let num_shares = 3;
        let x: Vec<Field64> = random_vector(len).unwrap();
        let mut x_shares = split_vector(&x, num_shares).unwrap();

        // Just for fun, let's do something different with a subset of the inputs. For the first
        // share, every odd element is set to the plaintext value. For all shares but the first,
        // every odd element is set to 0.
        #[allow(clippy::needless_range_loop)]
        for i in 0..len {
            if i % 2 != 0 {
                x_shares[0][i] = x[i];
            }
            for j in 1..num_shares {
                if i % 2 != 0 {
                    x_shares[j][i] = Field64::zero();
                }
            }
        }

        let mut got = vec![Field64::zero(); len];
        let mut buf = vec![Field64::zero(); len];
        for share in x_shares {
            discrete_fourier_transform_inv(&mut buf, &share, len).unwrap();
            for i in 0..len {
                got[i] += buf[i];
            }
        }

        let mut want = vec![Field64::zero(); len];
        discrete_fourier_transform_inv(&mut want, &x, len).unwrap();

        assert_eq!(got, want);
    }
}
