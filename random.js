
// Mersenne Twister 19937 — matches Python's `random` module exactly (for reproducibility)
const mt = new Uint32Array(624); let idx = 625; let _gauss_next = null;
function seed(n) {
    const u = (a, b) => Math.imul(a, b) >>> 0, key = [];
    for (let v = n || 0; v > 0; v = Math.floor(v / 0x100000000)) key.push(v & 0xFFFFFFFF);
    if (!key.length) key.push(0);
    mt[0] = 19650218; for (idx = 1; idx < 624; ++idx) mt[idx] = (u(1812433253, mt[idx-1] ^ (mt[idx-1] >>> 30)) + idx) >>> 0;
    let i = 1, j = 0;
    for (let k = Math.max(624, key.length); k > 0; --k, ++i, ++j) {
        if (i >= 624) { mt[0] = mt[623]; i = 1; } if (j >= key.length) j = 0;
        mt[i] = ((mt[i] ^ u(mt[i-1] ^ (mt[i-1] >>> 30), 1664525)) + key[j] + j) >>> 0;
    }
    for (let k = 623; k > 0; --k, ++i) {
        if (i >= 624) { mt[0] = mt[623]; i = 1; }
        mt[i] = ((mt[i] ^ u(mt[i-1] ^ (mt[i-1] >>> 30), 1566083941)) - i) >>> 0;
    }
    mt[0] = 0x80000000; idx = 624; _gauss_next = null;
}
function int32() {
    if (idx >= 624) { for (let k = 0; k < 624; ++k) { // twist
        const y = (mt[k] & 0x80000000) | (mt[(k+1) % 624] & 0x7FFFFFFF);
        mt[k] = (mt[(k+397) % 624] ^ (y >>> 1) ^ (y & 1 ? 0x9908B0DF : 0)) >>> 0;
    } idx = 0; }
    let y = mt[idx++];
    y ^= y >>> 11; y ^= (y << 7) & 0x9D2C5680; y ^= (y << 15) & 0xEFC60000; y ^= y >>> 18;
    return y >>> 0;
}
function random() { return ((int32() >>> 5) * 67108864.0 + (int32() >>> 6)) / 9007199254740992.0; }
function gauss(mu = 0, sigma = 1) { // Box-Muller with cached spare (matches Python)
    let z = _gauss_next; _gauss_next = null;
    if (z === null) { const x2pi = random() * 2 * Math.PI, g2rad = Math.sqrt(-2 * Math.log(1 - random()));
        z = Math.cos(x2pi) * g2rad; _gauss_next = Math.sin(x2pi) * g2rad; }
    return mu + z * sigma;
}
function shuffle(arr) { for (let i = arr.length - 1; i > 0; --i) { // Fisher-Yates via getrandbits (matches Python)
    const k = 32 - Math.clz32(i + 1); let r = int32() >>> (32 - k); while (r > i) r = int32() >>> (32 - k);
    const t = arr[i]; arr[i] = arr[r]; arr[r] = t;
} }
function choices(population, weights) { // bisect on cumulative weights (matches Python)
    const cum = new Float64Array(weights.length); cum[0] = weights[0];
    for (let i = 1; i < weights.length; ++i) cum[i] = cum[i-1] + weights[i];
    const x = random() * cum[cum.length - 1];
    let lo = 0, hi = cum.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; x < cum[mid] ? hi = mid : lo = mid + 1; }
    return population[lo];
}
export default { seed, random, gauss, shuffle, choices };
