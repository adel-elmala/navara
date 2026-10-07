import {
  GlyphCharClass,
  type GlyphMetrics,
  type ShapedGlyph,
  type ShapeTextResult,
} from "@navaramap/font";
import { describe, expect, it } from "vitest";

import {
  allowsLetterSpacing,
  breakLines,
  buildLabelLayout,
  isRtlText,
  lineWidthFu,
} from "./layout";

/** Build a glyph run from a compact spec: one entry per glyph. `cont` marks a
 *  glyph continuing the previous glyph's shaping cluster. */
function glyphs(
  spec: { advance?: number; cls?: number; cont?: boolean }[],
  defaultAdvance = 100,
): ShapedGlyph[] {
  return spec.map((s, i) => ({
    glyphId: i + 1,
    fontIndex: 0,
    compositeKey: BigInt(i + 1),
    xAdvance: s.advance ?? defaultAdvance,
    yAdvance: 0,
    xOffset: 0,
    yOffset: 0,
    charClass: s.cls ?? GlyphCharClass.Normal,
    continuesCluster: s.cont ?? false,
  }));
}

/** "ab cd" style shorthand: space → whitespace, "\n" → newline marker,
 *  "国" (any non-ASCII) → ideographic; everything else normal. */
function fromText(text: string, advance = 100): ShapedGlyph[] {
  return glyphs(
    [...text].map((ch) => ({
      advance: ch === "\n" ? 0 : advance,
      cls:
        ch === "\n"
          ? GlyphCharClass.Newline
          : ch === " "
            ? GlyphCharClass.Whitespace
            : ch.charCodeAt(0) > 127
              ? GlyphCharClass.Ideographic
              : GlyphCharClass.Normal,
    })),
  );
}

describe("breakLines", () => {
  it("keeps a run without breaks on a single line", () => {
    const lines = breakLines(fromText("abc"), 0);
    expect(lines.length).toBe(1);
    expect(lines[0].length).toBe(3);
  });

  it("splits at newline markers and drops the marker glyph", () => {
    const lines = breakLines(fromText("ab\ncd\nef"), 0);
    expect(lines.map((l) => l.length)).toEqual([2, 2, 2]);
    for (const line of lines) {
      expect(line.every((g) => g.charClass !== GlyphCharClass.Newline)).toBe(
        true,
      );
    }
  });

  it("preserves empty lines from consecutive newlines", () => {
    const lines = breakLines(fromText("a\n\nb"), 0);
    expect(lines.map((l) => l.length)).toEqual([1, 0, 1]);
  });

  it("does not wrap when maxWidth is 0", () => {
    const lines = breakLines(fromText("aa bb cc dd"), 0);
    expect(lines.length).toBe(1);
  });

  it("wraps at whitespace when a line exceeds maxWidth", () => {
    // Each glyph is 100 wide; "aa bb" fits in 500 but "aa bb cc" does not.
    const lines = breakLines(fromText("aa bb cc"), 500);
    expect(lines.length).toBe(2);
    // The wrap point's whitespace is dropped from both line ends.
    expect(lines[0].length).toBe(5); // "aa bb"
    expect(lines[1].length).toBe(2); // "cc"
  });

  it("drops the whitespace glyph at the wrap point", () => {
    const lines = breakLines(fromText("aa bb"), 300);
    expect(lines.length).toBe(2);
    expect(
      lines.flat().every((g) => g.charClass !== GlyphCharClass.Whitespace),
    ).toBe(true);
  });

  it("lets a word longer than maxWidth overflow instead of breaking mid-word", () => {
    const lines = breakLines(fromText("aaaaaa"), 300);
    expect(lines.length).toBe(1);
    expect(lines[0].length).toBe(6);
  });

  it("wraps after ideographic glyphs without whitespace", () => {
    const lines = breakLines(fromText("国国国国"), 250);
    expect(lines.length).toBe(2);
    expect(lines.map((l) => l.length)).toEqual([2, 2]);
  });

  it("combines hard breaks with soft wrapping", () => {
    const lines = breakLines(fromText("aa bb\ncc"), 300);
    expect(lines.map((l) => l.length)).toEqual([2, 2, 2]);
  });

  describe("rtl", () => {
    /** RTL glyph runs arrive in visual order = reversed logical order.
     *  Build from logical text, then reverse (per hard-break segment). */
    function rtlFromText(text: string, advance = 100): ShapedGlyph[] {
      const out: ShapedGlyph[] = [];
      let segment: ShapedGlyph[] = [];
      for (const g of fromText(text, advance)) {
        if (g.charClass === GlyphCharClass.Newline) {
          out.push(...segment.reverse(), g);
          segment = [];
        } else {
          segment.push(g);
        }
      }
      out.push(...segment.reverse());
      return out;
    }

    it("stacks wrapped lines in logical (reading) order, top to bottom", () => {
      // Logical "aa bb cc" with glyphIds 1..8; visual stream is reversed.
      const lines = breakLines(rtlFromText("aa bb cc"), 500, true);
      expect(lines.length).toBe(2);
      // Top line holds the logical start ("aa bb"), in visual order.
      expect(lines[0].map((g) => g.glyphId)).toEqual([5, 4, 3, 2, 1]);
      expect(lines[1].map((g) => g.glyphId)).toEqual([8, 7]);
    });

    it("fills lines greedily from the logical start", () => {
      // Five words, two per line: 2-2-1, not 1-2-2.
      const lines = breakLines(rtlFromText("a b c d e"), 300, true);
      expect(lines.map((l) => l.length)).toEqual([3, 3, 1]);
      expect(lines[2].map((g) => g.glyphId)).toEqual([9]); // logical last word
    });

    it("keeps hard-break segments in logical order", () => {
      const lines = breakLines(rtlFromText("aa\nbb"), 0, true);
      expect(lines.map((l) => l.map((g) => g.glyphId))).toEqual([
        [2, 1],
        [5, 4],
      ]);
    });

    it("matches LTR output for a single unwrapped line", () => {
      const lines = breakLines(rtlFromText("abc"), 0, true);
      expect(lines.length).toBe(1);
      expect(lines[0].map((g) => g.glyphId)).toEqual([3, 2, 1]);
    });
  });
});

describe("isRtlText", () => {
  it("detects Arabic", () => {
    expect(isRtlText("شارع الملك")).toBe(true);
  });

  it("detects Hebrew", () => {
    expect(isRtlText("רחוב")).toBe(true);
  });

  it("is false for Latin", () => {
    expect(isRtlText("Main St")).toBe(false);
  });

  it("is false for CJK", () => {
    expect(isRtlText("東京都")).toBe(false);
  });

  it("uses the first strong character in mixed text", () => {
    expect(isRtlText("Cafe شارع")).toBe(false);
    expect(isRtlText("شارع Cafe")).toBe(true);
  });

  it("skips leading digits and punctuation", () => {
    expect(isRtlText("12 - شارع")).toBe(true);
  });

  it("is false for empty or neutral-only text", () => {
    expect(isRtlText("")).toBe(false);
    expect(isRtlText("123 !?")).toBe(false);
  });
});

describe("lineWidthFu", () => {
  it("sums advances", () => {
    expect(lineWidthFu(fromText("abc"))).toBe(300);
  });

  it("ignores trailing whitespace", () => {
    expect(lineWidthFu(fromText("ab  "))).toBe(200);
  });

  it("counts interior whitespace", () => {
    expect(lineWidthFu(fromText("a b"))).toBe(300);
  });

  it("is 0 for an empty line", () => {
    expect(lineWidthFu([])).toBe(0);
  });
});

describe("buildLabelLayout word grouping", () => {
  /**
   * A shaping result for `text` where every visible character is a square
   * glyph one em wide. A space and a zero-width joiner both draw nothing; only
   * the space is classed as whitespace, which is what makes it — and not the
   * joiner — a word boundary.
   */
  function shaped(text: string): ShapeTextResult {
    const unitsPerEm = 1000;
    const glyphs: ShapedGlyph[] = [...text].map((ch, i) => ({
      glyphId: i + 1,
      fontIndex: 0,
      compositeKey: BigInt(i + 1),
      xAdvance: unitsPerEm,
      yAdvance: 0,
      xOffset: 0,
      yOffset: 0,
      charClass: ch === " " ? GlyphCharClass.Whitespace : GlyphCharClass.Normal,
    }));
    const metrics: GlyphMetrics[] = [...text].map((ch, i) => ({
      glyphId: i + 1,
      fontIndex: 0,
      compositeKey: BigInt(i + 1),
      atlasX: 0,
      atlasY: 0,
      // Neither has an atlas rectangle, so neither produces a quad.
      atlasW: ch === " " || ch === ZWJ ? 0 : 64,
      atlasH: ch === " " || ch === ZWJ ? 0 : 64,
      bearingX: 0,
      bearingY: 0,
      isColor: false,
    }));
    return {
      glyphs,
      metrics,
      unitsPerEm,
      ascender: 800,
      descender: -200,
      lineGap: 0,
    };
  }

  const options = {
    text: "",
    maxWidth: 0,
    lineHeight: 1,
    textAlign: 0,
    spreadGlyphs: false,
  };
  const ZWJ = "\u200D";

  it("keeps a word whole across a glyph that draws nothing", () => {
    // A join control inside a word has no quad, but it is not a space: the
    // word either side of it has to stay one rigid group, or a joined script
    // word would bend along the line as two separately rotated pieces.
    const text = `a${ZWJ}b cd`;
    const layout = buildLabelLayout(shaped(text), { ...options, text });
    expect(layout.quads.length).toBe(4);
    const centers = layout.quads.map((q) => q.wordCenterEmX);
    expect(centers[0]).toBe(centers[1]);
    expect(centers[2]).toBe(centers[3]);
    expect(centers[0]).not.toBe(centers[2]);
  });

  it("gives every glyph of a word the same centre", () => {
    const layout = buildLabelLayout(shaped("ab cd"), {
      ...options,
      text: "ab cd",
    });
    // Four drawn glyphs: the space contributes none.
    expect(layout.quads.length).toBe(4);
    const centers = layout.quads.map((q) => q.wordCenterEmX);
    expect(centers[0]).toBe(centers[1]);
    expect(centers[2]).toBe(centers[3]);
    expect(centers[0]).not.toBe(centers[2]);
  });

  it("places each word's centre at the middle of its own glyphs", () => {
    const layout = buildLabelLayout(shaped("ab cd"), {
      ...options,
      text: "ab cd",
    });
    // Glyphs advance one em each and are 64/64 = 1 em wide, so "ab" spans
    // [0, 2] and "cd" spans [3, 5] once the space has taken its em.
    expect(layout.quads[0].wordCenterEmX).toBeCloseTo(1, 5);
    expect(layout.quads[2].wordCenterEmX).toBeCloseTo(4, 5);
  });

  it("treats a single word as one group", () => {
    const layout = buildLabelLayout(shaped("abcd"), {
      ...options,
      text: "abcd",
    });
    const centers = new Set(layout.quads.map((q) => q.wordCenterEmX));
    expect(centers.size).toBe(1);
    expect([...centers][0]).toBeCloseTo(2, 5);
  });

  it("does not run a word across a line break", () => {
    // Wrapping puts "ab" and "cde" on their own lines, each starting at x = 0.
    // The words must be measured separately: spanning the break would give
    // every glyph the centre of all five together.
    const layout = buildLabelLayout(shaped("ab cde"), {
      ...options,
      text: "ab cde",
      maxWidth: 2,
    });
    expect(layout.quads.length).toBe(5);
    expect(layout.quads.slice(0, 2).map((q) => q.wordCenterEmX)).toEqual([
      1, 1,
    ]);
    expect(layout.quads.slice(2).map((q) => q.wordCenterEmX)).toEqual([
      1.5, 1.5, 1.5,
    ]);
  });

  it("splits words at a space that draws", () => {
    // U+1680 OGHAM SPACE MARK is whitespace with ink. It must still close the
    // word before it, and must not join the next one: otherwise both words
    // share one centre and turn on the line as a single rigid piece.
    const OGHAM_SPACE = " ";
    const text = `ab${OGHAM_SPACE}cd`;
    const result = shaped(text);
    result.glyphs[2].charClass = GlyphCharClass.Whitespace;
    const layout = buildLabelLayout(result, { ...options, text });
    // The space draws, so five quads: "ab" over [0, 2], the mark over [2, 3],
    // "cd" over [3, 5].
    expect(layout.quads.map((q) => q.wordCenterEmX)).toEqual([1, 1, 2.5, 4, 4]);
    expect(layout.maxWordHalfEm).toBeCloseTo(1, 5);
  });

  it("bounds the ink, not the advances", () => {
    // Glyph boxes are 1 em, as wide as their advances, unless a bearing moves
    // them: shifting the first a fifth of an em left and the last a fifth
    // right makes "abc" draw over [-0.2, 3.2] though it advances over [0, 3].
    const result = shaped("abc");
    result.metrics[0].bearingX = -0.2 * 64;
    result.metrics[2].bearingX = 0.2 * 64;
    const layout = buildLabelLayout(result, { ...options, text: "abc" });
    expect(layout.widthEm).toBeCloseTo(3, 5);
    expect(layout.minXEm).toBeCloseTo(-0.2, 5);
    expect(layout.maxXEm).toBeCloseTo(3.2, 5);
  });

  it("centres every glyph on itself with spreadGlyphs", () => {
    const layout = buildLabelLayout(shaped("ab cd"), {
      ...options,
      text: "ab cd",
      spreadGlyphs: true,
    });
    expect(layout.quads.map((q) => q.wordCenterEmX)).toEqual([
      0.5, 1.5, 3.5, 4.5,
    ]);
  });

  it("reaches half the widest piece along its tangent", () => {
    const words = buildLabelLayout(shaped("ab cde"), {
      ...options,
      text: "ab cde",
    });
    expect(words.maxWordHalfEm).toBeCloseTo(1.5, 5);

    const glyphs = buildLabelLayout(shaped("ab cde"), {
      ...options,
      text: "ab cde",
      spreadGlyphs: true,
    });
    expect(glyphs.maxWordHalfEm).toBeCloseTo(0.5, 5);
  });
});

describe("letter spacing", () => {
  describe("lineWidthFu", () => {
    it("adds spacing between glyphs but not after the last", () => {
      expect(lineWidthFu(fromText("abc"), 50)).toBe(400);
    });

    it("adds no spacing after trimmed trailing whitespace", () => {
      expect(lineWidthFu(fromText("ab  "), 50)).toBe(250);
    });

    it("keeps glyphs of one cluster together", () => {
      // base + zero-advance mark + next letter: one gap, not two.
      const run = glyphs([{}, { advance: 0, cont: true }, {}]);
      expect(lineWidthFu(run, 50)).toBe(250);
    });

    it("tightens with negative spacing", () => {
      expect(lineWidthFu(fromText("abc"), -20)).toBe(260);
    });
  });

  describe("breakLines", () => {
    it("counts spacing toward the wrap width", () => {
      // "aa bb" is exactly 500 without spacing; four gaps push it over.
      expect(breakLines(fromText("aa bb"), 500, false, 0).length).toBe(1);
      expect(breakLines(fromText("aa bb"), 500, false, 10).length).toBe(2);
    });

    it("measures RTL clusters the same as the final visual layout", () => {
      // Logical "x ab" where `a` carries a mark. Visual (shaper) order is
      // b, mark, base, space, x — the base continues the mark's cluster.
      const visual = glyphs([
        {}, // b
        { advance: 0 }, // mark (cluster start in visual order)
        { cont: true }, // base
        { cls: GlyphCharClass.Whitespace },
        {}, // x
      ]);
      // 400 advance + three inter-cluster gaps = 550: fits exactly.
      const lines = breakLines(visual, 550, true, 50);
      expect(lines.length).toBe(1);
      expect(lineWidthFu(lines[0], 50)).toBe(550);
    });
  });

  describe("allowsLetterSpacing", () => {
    it("allows Latin, CJK and Hebrew", () => {
      expect(allowsLetterSpacing("Main St")).toBe(true);
      expect(allowsLetterSpacing("東京都")).toBe(true);
      expect(allowsLetterSpacing("רחוב")).toBe(true);
    });

    it("disallows any Arabic-script text", () => {
      expect(allowsLetterSpacing("شارع")).toBe(false);
      expect(allowsLetterSpacing("Cafe شارع")).toBe(false);
    });
  });

  describe("buildLabelLayout", () => {
    /** Wrap a glyph run in a shaping result whose em is 1000 units and whose
     *  glyphs all have a drawable 10×10 atlas rect with zero bearing, so a
     *  quad's `offsetEmX` is exactly its pen position / 1000. */
    function shaped(run: ShapedGlyph[]): ShapeTextResult {
      return {
        glyphs: run,
        metrics: run.map((g) => ({
          glyphId: g.glyphId,
          fontIndex: 0,
          compositeKey: g.compositeKey,
          atlasX: 0,
          atlasY: 0,
          atlasW: 10,
          atlasH: 10,
          bearingX: 0,
          bearingY: 0,
          isColor: false,
        })),
        unitsPerEm: 1000,
        ascender: 800,
        descender: -200,
        lineGap: 0,
      };
    }

    const base = {
      maxWidth: 0,
      lineHeight: 1,
      textAlign: 0,
      spreadGlyphs: false,
    };

    it("widens the block by (n - 1) gaps", () => {
      const run = shaped(fromText("abc"));
      const plain = buildLabelLayout(run, { ...base, text: "abc" });
      const spaced = buildLabelLayout(run, {
        ...base,
        text: "abc",
        letterSpacing: 0.1,
      });
      expect(plain.widthEm).toBeCloseTo(0.3);
      expect(spaced.widthEm).toBeCloseTo(0.5);
      expect(spaced.quads.map((q) => q.offsetEmX)).toEqual([
        expect.closeTo(0),
        expect.closeTo(0.2),
        expect.closeTo(0.4),
      ]);
    });

    it("centers lines using the spaced widths", () => {
      const layout = buildLabelLayout(shaped(fromText("ab\nabcd")), {
        ...base,
        text: "ab\nabcd",
        textAlign: 0.5,
        letterSpacing: 0.1,
      });
      // Block 0.7 em, short line 0.3 em: it starts 0.2 em in.
      expect(layout.widthEm).toBeCloseTo(0.7);
      expect(layout.quads[0].offsetEmX).toBeCloseTo(0.2);
    });

    it("ignores spacing for Arabic text", () => {
      const layout = buildLabelLayout(shaped(fromText("abc")), {
        ...base,
        text: "شارع",
        letterSpacing: 0.1,
      });
      expect(layout.widthEm).toBeCloseTo(0.3);
    });
  });
});
