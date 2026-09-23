import { MathUtils } from "three";

/**
 * Packing for the Rust `lineLabelPlace` kernel.
 *
 * Kept separate from the mesh, and free of three.js and WASM, so the layout
 * contract with `crates/navara_wasm_api/src/line_label.rs` is unit-testable on
 * its own — the same split `declutter/kernel.ts` makes for `declutterPlace`.
 */

/** `f64` values per label in the kernel's packed input. */
export const LINE_LABEL_STRIDE = 17;

/** `f64` values per label in the kernel's packed output: flip, rejected, then
 *  the rotated box as minX/maxX/minY/maxY. */
export const LINE_LABEL_RESULT_STRIDE = 6;

/** `f64` values per label in the `lineLabelFit` pre-pass's packed input. */
export const LINE_LABEL_FIT_STRIDE = 7;

/** Scalars the engine sends per anchor alongside its path samples. */
export const PATH_META_STRIDE = 2;

/** The subset of a label record this packing reads. */
export type PackableLabel = {
  slot: number;
  instanceIndex: number;
  anchor: Float64Array;
  addHeight: number;
  widthEm: number;
  heightEm: number;
  minYEm: number;
  maxYEm: number;
  fontSize: number;
};

/** The resampled line each anchor sits on. */
export type LinePath = {
  /** East/north metre offsets from each anchor, `stride` floats per anchor. */
  samples: Float32Array<ArrayBufferLike>;
  /** Floats per anchor — twice the sample count. */
  stride: number;
  /** Per-anchor `(metres between samples, metres of real line either side)`. */
  meta: Float32Array<ArrayBufferLike> | null;
  /** Per-anchor tangent bearing in degrees clockwise from north. */
  bearings: Float32Array<ArrayBufferLike> | null;
};

export type LinePlacementOptions = {
  sizeInMeters: boolean;
  maxAngleDeg: number;
  keepUpright: boolean;
  /** The text block's anchor point, already clamped to [-0.5, 0.5]. */
  center: readonly [number, number];
  /** Whether the label is currently walked backwards, which feeds the kernel's
   *  flip hysteresis. */
  readFlip: (slot: number) => boolean;
};

/**
 * Flatten labels into the fit pre-pass's compact input.
 *
 * Deliberately carries no path samples: whether a label is short enough to sit
 * on its line depends only on the anchor, the label's width and the length of
 * line under it. Those are 7 scalars against the 32 path points the full pass
 * needs, and on a dense view most labels fail this test — so running it first,
 * over this array, keeps the large payload off the boundary for the labels that
 * were never going to be placed.
 *
 * Field order must match `LINE_LABEL_FIT_STRIDE`'s table in `line_label.rs`.
 */
export function packLineLabelFits(
  labels: readonly PackableLabel[],
  path: LinePath,
  sizeInMeters: boolean,
): Float64Array {
  const out = new Float64Array(labels.length * LINE_LABEL_FIT_STRIDE);
  const meta = path.meta;
  const metric = sizeInMeters ? 1 : 0;
  for (let i = 0; i < labels.length; i++) {
    const label = labels[i];
    const o = i * LINE_LABEL_FIT_STRIDE;
    out[o] = label.anchor[0];
    out[o + 1] = label.anchor[1];
    out[o + 2] = label.anchor[2];
    out[o + 3] = label.widthEm;
    out[o + 4] = label.fontSize;
    out[o + 5] = metric;
    out[o + 6] = meta?.[label.instanceIndex * PATH_META_STRIDE + 1] ?? 0;
  }
  return out;
}

/**
 * Flatten labels and their path samples into the kernel's two input arrays.
 *
 * Field order must match the table in `line_label.rs`; the two move together.
 */
export function packLineLabels(
  labels: readonly PackableLabel[],
  path: LinePath,
  options: LinePlacementOptions,
): { labels: Float64Array; paths: Float32Array } {
  const n = labels.length;
  const stride = path.stride;
  const out = new Float64Array(n * LINE_LABEL_STRIDE);
  const paths = new Float32Array(n * stride);
  const samples = path.samples;
  const meta = path.meta;
  const bearings = path.bearings;
  const maxAngleRad = MathUtils.degToRad(options.maxAngleDeg);

  for (let i = 0; i < n; i++) {
    const label = labels[i];
    const o = i * LINE_LABEL_STRIDE;
    out[o] = label.anchor[0];
    out[o + 1] = label.anchor[1];
    out[o + 2] = label.anchor[2];
    out[o + 3] = label.addHeight;
    out[o + 4] = label.widthEm;
    out[o + 5] = label.fontSize;
    out[o + 6] = options.sizeInMeters ? 1 : 0;
    out[o + 7] = maxAngleRad;
    out[o + 8] = options.keepUpright ? 1 : 0;
    out[o + 9] = meta?.[label.instanceIndex * PATH_META_STRIDE] ?? 0;
    out[o + 10] = meta?.[label.instanceIndex * PATH_META_STRIDE + 1] ?? 0;
    out[o + 11] = MathUtils.degToRad(bearings?.[label.instanceIndex] ?? 0);
    out[o + 12] = options.readFlip(label.slot) ? 1 : 0;

    // The unrotated collision box, in the font's own units. Computed here
    // rather than in Rust so the em-and-anchor arithmetic stays in the one
    // place that also builds the box for point labels.
    const [cx, cy] = options.center;
    const w = label.widthEm;
    const h = label.heightEm;
    out[o + 13] = (0 - cx * w) * label.fontSize;
    out[o + 14] = (w - cx * w) * label.fontSize;
    out[o + 15] = (label.minYEm - cy * h) * label.fontSize;
    out[o + 16] = (label.maxYEm - cy * h) * label.fontSize;

    // Labels are created lazily and sparsely, so their path runs are gathered
    // into input order rather than passed as one contiguous slice.
    paths.set(
      samples.subarray(
        label.instanceIndex * stride,
        (label.instanceIndex + 1) * stride,
      ),
      i * stride,
    );
  }

  return { labels: out, paths };
}

/**
 * Lift the line path out of a freshly extracted geometry, or `null` when that
 * geometry carries none.
 *
 * Separated from the position info because the two have different lifetimes: a
 * geometry update replaces the positions on every terrain height change, while
 * the path is fixed for the tile.
 */
export function takeLinePath(
  info: {
    pathSamples: Float32Array<ArrayBufferLike> | null;
    pathStride: number;
    pathMeta: Float32Array<ArrayBufferLike> | null;
    bearings: Float32Array<ArrayBufferLike> | null;
  } | null,
): LinePath | null {
  if (!info?.pathSamples || info.pathStride <= 0) return null;
  return {
    samples: info.pathSamples,
    stride: info.pathStride,
    meta: info.pathMeta,
    bearings: info.bearings,
  };
}
