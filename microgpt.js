/**
 * The most atomic way to train and inference a GPT in pure, dependency-free JavaScript.
 * This file is the complete algorithm.
 * Everything else is just efficiency.
 *
 * @karpathy (original Python), @xenova (JavaScript port)
 */

import fs from 'fs'; // for reading the input text file
import random from './random.js'; // random.seed, random.choices, random.gauss, random.shuffle
random.seed(42); // Let there be order among chaos

const docs = fs.readFileSync('input.txt', 'utf-8').trim().split('\n').map(l => l.trim()).filter(l => l.length > 0); // list of documents
random.shuffle(docs);
console.log(`num docs: ${docs.length}`);

// Let there be a Tokenizer to translate strings to discrete symbols and back
const uchars = [...new Set(docs.join(''))].sort(); // unique characters in the dataset become token ids 0..n-1
const char_to_id = new Map(uchars.map((ch, i) => [ch, i])); // fast character lookup
const BOS = uchars.length; // token id for the special Beginning of Sequence (BOS) token
const vocab_size = uchars.length + 1; // total number of unique tokens, +1 is for BOS
console.log(`vocab size: ${vocab_size}`);

// Let there be Autograd, to recursively apply the chain rule through a computation graph
let _gen = 0; // global generation counter for autograd, to help with topological sorting of the graph during backward pass
class Value {
  constructor(data, children = [], local_grads = []) {
    this.data = data;             // scalar value of this node calculated during forward pass
    this.grad = 0;                // derivative of the loss w.r.t. this node, calculated in backward pass
    this._c0 = children[0];       // children of this node in the computation graph
    this._c1 = children[1];
    this._lg0 = local_grads[0];   // local derivative of this node w.r.t. its children
    this._lg1 = local_grads[1];
    this._nch = children.length;  // number of children (0, 1, or 2)
    this._gen = 0;
  }

  add(other) {
    if (other instanceof Value) return new Value(this.data + other.data, [this, other], [1, 1]);
    return new Value(this.data + other, [this], [1]);
  }

  mul(other) {
    if (other instanceof Value) return new Value(this.data * other.data, [this, other], [other.data, this.data]);
    return new Value(this.data * other, [this], [other]);
  }

  pow(other) { return new Value(this.data ** other, [this], [other * this.data ** (other - 1)]); }
  log() { return new Value(Math.log(this.data), [this], [1 / this.data]); }
  exp() { const e = Math.exp(this.data); return new Value(e, [this], [e]); }
  relu() { return new Value(Math.max(0, this.data), [this], [+(this.data > 0)]); }
  neg() { return new Value(-this.data, [this], [-1]); }
  sub(other) { return this.add(other instanceof Value ? other.neg() : -other); }
  div(other) { return this.mul(other instanceof Value ? other.pow(-1) : 1 / other); }

  backward() {
    const gen = ++_gen;
    const topo = [];
    function build_topo(v) {
      if (v._gen === gen) return;
      v._gen = gen;
      if (v._nch >= 1) build_topo(v._c0);
      if (v._nch === 2) build_topo(v._c1);
      topo.push(v);
    }
    build_topo(this);
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; --i) {
      const v = topo[i], g = v.grad;
      if (v._nch >= 1) v._c0.grad += v._lg0 * g;
      if (v._nch === 2) v._c1.grad += v._lg1 * g;
    }
  }
}

// Initialize the parameters, to store the knowledge of the model.
const n_embd = 16;     // embedding dimension
const n_head = 4;      // number of attention heads
const n_layer = 1;     // number of layers
const block_size = 16; // maximum sequence length
const head_dim = Math.floor(n_embd / n_head); // dimension of each head
const scale = 1 / head_dim ** 0.5; // precomputed attention scale factor
const matrix = (nout, nin, std = 0.08) => Array.from({ length: nout }, () => Array.from({ length: nin }, () => new Value(random.gauss(0, std))));
const state_dict = { wte: matrix(vocab_size, n_embd), wpe: matrix(block_size, n_embd), lm_head: matrix(vocab_size, n_embd) };
for (let i = 0; i < n_layer; ++i) {
  state_dict[`layer${i}.attn_wq`] = matrix(n_embd, n_embd);
  state_dict[`layer${i}.attn_wk`] = matrix(n_embd, n_embd);
  state_dict[`layer${i}.attn_wv`] = matrix(n_embd, n_embd);
  state_dict[`layer${i}.attn_wo`] = matrix(n_embd, n_embd);
  state_dict[`layer${i}.mlp_fc1`] = matrix(4 * n_embd, n_embd);
  state_dict[`layer${i}.mlp_fc2`] = matrix(n_embd, 4 * n_embd);
}
const params = Object.values(state_dict).flat(Infinity); // flatten params into a single list of Values
console.log(`num params: ${params.length}`);

// Define the model architecture: a stateless function mapping token sequence and parameters to logits over what comes next.
// Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
const sum = (arr) => arr.reduce((a, b) => a.add(b));
const zip = (a, b) => a.map((ai, i) => [ai, b[i]]);

function linear(x, w) {
  return w.map(wo => sum(wo.map((wi, i) => wi.mul(x[i]))));
}

function softmax(logits) {
  const max_val = Math.max(...logits.map(v => v.data));
  const exps = logits.map(v => v.sub(max_val).exp());
  const total = sum(exps);
  return exps.map(e => e.div(total));
}

function rmsnorm(x) {
  const ms = sum(x.map(xi => xi.mul(xi))).mul(1 / x.length);
  const s = ms.add(1e-5).pow(-0.5);
  return x.map(xi => xi.mul(s));
}

function gpt(token_id, pos_id, keys, values) {
  const tok_emb = state_dict['wte'][token_id]; // token embedding
  const pos_emb = state_dict['wpe'][pos_id];   // position embedding
  let x = zip(tok_emb, pos_emb).map(([t, p]) => t.add(p)); // joint token and position embedding
  x = rmsnorm(x);

  for (let li = 0; li < n_layer; ++li) {
    // 1) Multi-head attention block
    let x_residual = x;
    x = rmsnorm(x);
    const q = linear(x, state_dict[`layer${li}.attn_wq`]);
    const k = linear(x, state_dict[`layer${li}.attn_wk`]);
    const v = linear(x, state_dict[`layer${li}.attn_wv`]);
    keys[li].push(k);
    values[li].push(v);
    const x_attn = [];
    for (let h = 0; h < n_head; ++h) {
      const hs = h * head_dim;
      const q_h = q.slice(hs, hs + head_dim);
      const k_h = keys[li].map(ki => ki.slice(hs, hs + head_dim));
      const v_h = values[li].map(vi => vi.slice(hs, hs + head_dim));
      const attn_logits = k_h.map(kt => sum(zip(q_h, kt).map(([qi, ki]) => qi.mul(ki))).mul(scale));
      const attn_weights = softmax(attn_logits);
      for (let j = 0; j < head_dim; ++j)
        x_attn.push(sum(attn_weights.map((aw, t) => aw.mul(v_h[t][j]))));
    }
    x = linear(x_attn, state_dict[`layer${li}.attn_wo`]);
    x = x.map((a, i) => a.add(x_residual[i]));
    // 2) MLP block
    x_residual = x;
    x = rmsnorm(x);
    x = linear(x, state_dict[`layer${li}.mlp_fc1`]);
    x = x.map(xi => xi.relu());
    x = linear(x, state_dict[`layer${li}.mlp_fc2`]);
    x = x.map((a, i) => a.add(x_residual[i]));
  }

  return linear(x, state_dict['lm_head']);
}

// Let there be Adam, the blessed optimizer and its buffers
const learning_rate = 0.01, beta1 = 0.85, beta2 = 0.99, eps_adam = 1e-8;
const m_buf = new Float64Array(params.length); // first moment buffer
const v_buf = new Float64Array(params.length); // second moment buffer

// Repeat in sequence
const num_steps = 1000; // number of training steps
for (let step = 0; step < num_steps; ++step) {

  // Take single document, tokenize it, surround it with BOS special token on both sides
  const doc = docs[step % docs.length];
  const tokens = [BOS, ...Array.from(doc, ch => char_to_id.get(ch)), BOS];
  const n = Math.min(block_size, tokens.length - 1);

  // Forward the token sequence through the model, building up the computation graph all the way to the loss.
  const keys = Array.from({ length: n_layer }, () => []);
  const values = Array.from({ length: n_layer }, () => []);
  const losses = [];
  for (let pos_id = 0; pos_id < n; ++pos_id) {
    const token_id = tokens[pos_id], target_id = tokens[pos_id + 1];
    const logits = gpt(token_id, pos_id, keys, values);
    const probs = softmax(logits);
    const loss_t = probs[target_id].log().neg();
    losses.push(loss_t);
  }
  const loss = sum(losses).mul(1 / n); // final average loss over the document sequence. May yours be low.

  // Backward the loss, calculating the gradients with respect to all model parameters.
  loss.backward();

  // Adam optimizer update: update the model parameters based on the corresponding gradients.
  const lr_t = learning_rate * (1 - step / num_steps); // linear learning rate decay
  const bc1 = 1 - beta1 ** (step + 1), bc2 = 1 - beta2 ** (step + 1);
  for (let i = 0; i < params.length; ++i) {
    const p = params[i];
    m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad;
    v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2;
    const m_hat = m_buf[i] / bc1;
    const v_hat = v_buf[i] / bc2;
    p.data -= lr_t * m_hat / (Math.sqrt(v_hat) + eps_adam);
    p.grad = 0;
  }

  console.log(`step ${String(step + 1).padStart(4)} / ${String(num_steps).padStart(4)} | loss ${loss.data.toFixed(4)}`);
}

// Inference: may the model babble back to us
const temperature = 0.5; // in (0, 1], control the "creativity" of generated text, low to high
const token_ids = Array.from({ length: vocab_size }, (_, i) => i);
console.log('\n--- inference (new, hallucinated names) ---');
for (let sample_idx = 0; sample_idx < 20; ++sample_idx) {
  const keys = Array.from({ length: n_layer }, () => []);
  const values = Array.from({ length: n_layer }, () => []);
  let token_id = BOS;
  const sample = [];
  for (let pos_id = 0; pos_id < block_size; ++pos_id) {
    const logits = gpt(token_id, pos_id, keys, values);
    const probs = softmax(logits.map(l => l.div(temperature)));
    token_id = random.choices(token_ids, probs.map(p => p.data));
    if (token_id === BOS) break;
    sample.push(uchars[token_id]);
  }
  console.log(`sample ${String(sample_idx + 1).padStart(2)}: ${sample.join('')}`);
}
