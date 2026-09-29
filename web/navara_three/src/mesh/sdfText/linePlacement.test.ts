import { describe, expect, it } from "vitest";

import {
  LINE_LABEL_FIT_STRIDE,
  LINE_LABEL_STRIDE,
  PATH_META_STRIDE,
  type LinePath,
  type PackableLabel,
  packLineLabelFits,
  packLineLabels,
  takeLinePath,
} from "./linePlacement";

/** 32 samples per anchor, matching `PATH_SAMPLES` in line_placement.rs. */
const SAMPLES = 32;

/**
 * One anchor's path. `stepMeters` and `realLineMeters` are the two scalars the
 * engine sends alongside the samples; the sample values themselves are never
 * read by the packing under test.
 */
function path(stepMeters: number, realLineMeters: number): LinePath {
  const meta = new Float32Array(PATH_META_STRIDE);
  meta[0] = stepMeters;
  meta[1] = realLineMeters;
  return {
    samples: new Float32Array(SAMPLES * 2),
    stride: SAMPLES * 2,
    meta,
    bearings: new Float32Array([0]),
  };
}

function label(over: Partial<PackableLabel> = {}): PackableLabel {
  return {
    slot: 0,
    instanceIndex: 0,
    anchor: new Float64Array([1, 2, 3]),
    addHeight: 0,
    widthEm: 4,
    heightEm: 1,
    minYEm: 0,
    maxYEm: 1,
    fontSize: 10,
    ...over,
  };
}

const options = {
  sizeInMeters: true,
  maxAngleDeg: 45,
  keepUpright: true,
  center: [0.5, 0] as const,
  lineOffset: 0,
  readFlip: () => false,
};

/** The span the samples cover either side of the anchor. */
const spanFor = (stepMeters: number) => 0.5 * (SAMPLES - 1) * stepMeters;

describe("usable half extent", () => {
  // A label longer than the sampled path does not fail gracefully: the vertex
  // shader clamps its lookup to the last sample, so the far end of the label
  // piles onto that one point and its words overlap. The extent handed to the
  // kernel therefore has to be capped by the span, not just by the road.
  it("caps the extent at the sampled span when the road is longer", () => {
    const step = 2;
    const span = spanFor(step); // 31 m
    const p = path(step, 10_000); // road far longer than the samples cover

    const fit = packLineLabelFits([label()], p, options);
    expect(fit[7]).toBeCloseTo(span, 5);

    const full = packLineLabels([label()], p, options);
    expect(full.labels[10]).toBeCloseTo(span, 5);
  });

  it("keeps the road limit when the road is the shorter of the two", () => {
    const step = 100;
    const p = path(step, 12); // only 12 m of road, samples cover 1550 m

    expect(packLineLabelFits([label()], p, options)[7]).toBeCloseTo(12, 5);
    expect(packLineLabels([label()], p, options).labels[10]).toBeCloseTo(12, 5);
  });

  it("gives both packings the same extent for the same anchor", () => {
    // The two phases must agree about what fits, or the cheap pre-pass would
    // admit labels the full pass then rejects — or worse, the other way round.
    for (const [step, road] of [
      [2, 10_000],
      [100, 12],
      [4, 60],
      [0, 500],
    ]) {
      const p = path(step, road);
      expect(packLineLabelFits([label()], p, options)[7]).toBe(
        packLineLabels([label()], p, options).labels[10],
      );
    }
  });

  it("reads the meta of the label's own anchor", () => {
    // Labels are created sparsely, so a label's `instanceIndex` addresses its
    // run in the shared meta array rather than its position in the input.
    const meta = new Float32Array(PATH_META_STRIDE * 3);
    meta[0] = 100;
    meta[1] = 5; // anchor 0: short road
    meta[PATH_META_STRIDE * 2] = 100;
    meta[PATH_META_STRIDE * 2 + 1] = 7; // anchor 2: slightly longer
    const p: LinePath = {
      samples: new Float32Array(SAMPLES * 2 * 3),
      stride: SAMPLES * 2,
      meta,
      bearings: new Float32Array([0, 0, 0]),
    };

    const fit = packLineLabelFits([label({ instanceIndex: 2 })], p, options);
    expect(fit[7]).toBeCloseTo(7, 5);
  });

  it("treats a missing meta array as no usable line", () => {
    const p: LinePath = {
      samples: new Float32Array(SAMPLES * 2),
      stride: SAMPLES * 2,
      meta: null,
      bearings: null,
    };
    expect(packLineLabelFits([label()], p, options)[7]).toBe(0);
  });
});

describe("packing shape", () => {
  it("writes one stride per label in input order", () => {
    const p = path(10, 500);
    const labels = [label({ slot: 0 }), label({ slot: 1, widthEm: 9 })];

    const fit = packLineLabelFits(labels, p, {
      ...options,
      sizeInMeters: false,
    });
    expect(fit.length).toBe(labels.length * LINE_LABEL_FIT_STRIDE);
    // Centred, so each label reaches half its width either side.
    expect(fit[4]).toBe(2);
    expect(fit[LINE_LABEL_FIT_STRIDE + 4]).toBe(4.5);

    const full = packLineLabels(labels, p, options);
    expect(full.labels.length).toBe(labels.length * LINE_LABEL_STRIDE);
    expect(full.paths.length).toBe(labels.length * p.stride);
  });
});

describe("anchor and offset", () => {
  // The shader lays text over [-cx·w, (1 - cx)·w] around the anchor, so an
  // off-centre anchor leaves one side needing more line than half the width.
  it("fits the longer side of an off-centre label", () => {
    const p = path(10, 500);
    const at = (cx: number) => ({ ...options, center: [cx, 0] as const });
    for (const [cx, reach] of [
      [0.5, 2],
      [0, 4],
      [-0.5, 6],
    ]) {
      expect(packLineLabelFits([label()], p, at(cx))[4]).toBe(reach);
      expect(packLineLabels([label()], p, at(cx)).labels[4]).toBe(reach);
    }
  });

  it("sends the anchor's height to both phases", () => {
    const p = path(10, 500);
    const raised = label({ addHeight: 120 });
    expect(packLineLabelFits([raised], p, options)[3]).toBe(120);
    expect(packLineLabels([raised], p, options).labels[3]).toBe(120);
  });

  it("moves the collision box with the line offset", () => {
    const p = path(10, 500);
    const plain = packLineLabels([label()], p, options).labels;
    const lifted = packLineLabels([label()], p, {
      ...options,
      lineOffset: 7,
    }).labels;
    expect(lifted[15]).toBe(plain[15] + 7);
    expect(lifted[16]).toBe(plain[16] + 7);
    expect(lifted[13]).toBe(plain[13]);
    expect(lifted[14]).toBe(plain[14]);
  });
});

describe("takeLinePath", () => {
  it("returns null when the geometry carries no path", () => {
    expect(takeLinePath(null)).toBeNull();
    expect(
      takeLinePath({
        pathSamples: null,
        pathStride: 0,
        pathMeta: null,
        bearings: null,
      }),
    ).toBeNull();
  });

  it("lifts the path out when one is present", () => {
    const samples = new Float32Array(4);
    const lifted = takeLinePath({
      pathSamples: samples,
      pathStride: 4,
      pathMeta: null,
      bearings: null,
    });
    expect(lifted?.samples).toBe(samples);
    expect(lifted?.stride).toBe(4);
  });
});
