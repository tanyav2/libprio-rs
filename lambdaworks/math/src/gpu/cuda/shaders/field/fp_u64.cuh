#ifndef felt_u64_h
#define felt_u64_h

// Using a simple uint64_t to represent values in the field
using u64 = unsigned long long;

class Fp64 {
public:
    Fp64() = default;
    __device__ constexpr Fp64(u64 v) : inner(v % N) {}
    
    __device__ constexpr explicit operator u64() const { return inner; }

    __device__ constexpr Fp64 operator+(const Fp64 rhs) const {
        return Fp64((inner + rhs.inner) % N);
    }

    __device__ constexpr Fp64 operator-(const Fp64 rhs) const {
        return Fp64((N + inner - rhs.inner) % N);
    }

    __device__ Fp64 operator*(const Fp64 rhs) const {
        return Fp64((inner * rhs.inner) % N);
    }

    __device__ Fp64 inverse() const {
        // Simple binary exponentiation for modular inverse.
        // This method is efficient enough for a 64-bit field.
        return pow(N - 2);
    }

    __device__ Fp64 pow(u64 exp) const {
        Fp64 base = *this;
        Fp64 result(1);
        while (exp) {
            if (exp & 1) {
                result = result * base;
            }
            base = base * base;
            exp >>= 1;
        }
        return result;
    }

    __device__ Fp64 neg() const {
        return Fp64(N - inner);
    }

private:
    u64 inner;

    // Modulus for the field.
    constexpr static const u64 N = 18446744069414584321ULL;
};

namespace p64 {
    using Fp = Fp64;
}

#endif
