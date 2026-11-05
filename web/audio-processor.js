// AudioWorklet: microphone at the context's rate -> 16 kHz mono Float32
// chunks of ~100 ms, posted to the main thread with their RMS level.
//
// Resampling is a box filter over each output interval (an average of the
// input samples it covers, fractional edges weighted), which is a cheap
// anti-alias for the 44.1/48 kHz -> 16 kHz decimation speech needs.

const TARGET_RATE = 16000;
const CHUNK = 1600;

class Downsampler extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / TARGET_RATE; // input samples per output sample
    this.acc = 0;      // weighted sum for the current output sample
    this.fill = 0;     // input samples' weight collected so far
    this.out = new Float32Array(CHUNK);
    this.n = 0;
    this.port.onmessage = (e) => {
      if (e.data === "flush") this.flush();
    };
  }

  flush() {
    if (this.n > 0) {
      const chunk = this.out.slice(0, this.n);
      this.post(chunk);
      this.n = 0;
    }
    this.port.postMessage({ type: "flushed" });
  }

  post(chunk) {
    let sq = 0;
    for (let i = 0; i < chunk.length; i++) sq += chunk[i] * chunk[i];
    this.port.postMessage(
      { type: "audio", samples: chunk, rms: Math.sqrt(sq / chunk.length) },
      [chunk.buffer],
    );
  }

  emit(v) {
    this.out[this.n++] = v;
    if (this.n === CHUNK) {
      this.post(this.out);
      this.out = new Float32Array(CHUNK);
      this.n = 0;
    }
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const chans = input.length;
    const len = input[0].length;
    for (let i = 0; i < len; i++) {
      let x = 0;
      for (let c = 0; c < chans; c++) x += input[c][i];
      x /= chans;
      // This input sample covers [0, 1) of weight; split it across the
      // output interval boundary when it straddles one.
      let w = 1;
      while (this.fill + w >= this.ratio) {
        const take = this.ratio - this.fill;
        this.acc += x * take;
        this.emit(this.acc / this.ratio);
        w -= take;
        this.acc = 0;
        this.fill = 0;
      }
      this.acc += x * w;
      this.fill += w;
    }
    return true;
  }
}

registerProcessor("downsampler", Downsampler);
