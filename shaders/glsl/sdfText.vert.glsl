#include "chunks/horizon_culling_pars_vertex.glsl"
#include "chunks/sprite_height_pars_vertex.glsl"
#include "chunks/pixelToWorld.glsl"
#include "chunks/quad_orientation.glsl"

// Glyph instances have no `_batchid` attribute; the feature index rides in
// the label data texture's STATE row instead (read into this local before
// the batch_texture_vertex include).
#define NVR_BATCH_ID_EXPR nvr_labelBatchIndex
#include "chunks/batch_texture_pars_vertex.glsl"

// One draw call covers every label in a tile-layer batch. Instances are
// GLYPHS, not labels, so anything that varies per label is read from
// uLabelData through the per-instance `labelIndex` rather than being a
// uniform. See web/navara_three/src/mesh/sdfText/labelData.ts — LABEL_ROWS and
// the row indices below must match `LabelRow` there (LABEL_ROWS is injected as
// a define from that same constant).

// Per-instance attributes
attribute vec2 glyphOffset;  // Glyph position in normalized text space
attribute vec2 glyphSize;    // Glyph quad dimensions in normalized text space
attribute vec4 glyphUvRect;  // Atlas sub-rect in PIXEL space: (x0, y0, x1, y1)
attribute float glyphKind;   // See GLYPH_KIND_* below
attribute float labelIndex;  // Row block in uLabelData owning this instance
#ifdef NVR_LINE_PLACEMENT
// Centre in x, in ems, of the word this glyph belongs to — shared by every
// glyph in that word. A curved label follows its line word by word, not letter
// by letter; see the walk below.
attribute float glyphWordCenter;
#endif

#define GLYPH_KIND_SDF        0.0
#define GLYPH_KIND_COLOR      1.0
#define GLYPH_KIND_BACKGROUND 2.0
// A slot inside a label's run that its current text doesn't use, or a hole
// left by a freed run. Culled outright.
#define GLYPH_KIND_EMPTY      3.0

#define LABEL_ROW_POSITION_HIGH 0
#define LABEL_ROW_POSITION_LOW 1
#define LABEL_ROW_BOX 2
#define LABEL_ROW_STATE 3
#define LABEL_ROW_PATH 4

// Per-label data texture (RGBA32F, unfiltered).
uniform sampler2D uLabelData;
uniform ivec2 uLabelTexSize;

vec4 nvr_readLabel(int slot, int row) {
    int i = slot * LABEL_ROWS + row;
    return texelFetch(uLabelData, ivec2(i % uLabelTexSize.x, i / uLabelTexSize.x), 0);
}

#ifdef NVR_LINE_PLACEMENT
// Each label's line, resampled at a uniform arc-length step as east/north
// metre offsets from its anchor. Uniform spacing is the whole point: a glyph
// finds its segment with one division instead of walking the path, so bending
// costs two texel fetches rather than a loop. PATH_SAMPLES is injected as a
// define from the same Rust constant that produced the data.
uniform sampler2D uPathData;
uniform ivec2 uPathTexSize;
// Perpendicular shift away from the line, in the same units as the font size
// (pixels or metres, per uSizeInMeters). Positive is left of travel.
uniform float uLineOffset;

// Two samples per RGBA texel, so a label's run is PATH_SAMPLES/2 texels.
vec2 nvr_readPath(int base, int k) {
    int i = base + (k >> 1);
    vec4 texel = texelFetch(uPathData, ivec2(i % uPathTexSize.x, i / uPathTexSize.x), 0);
    return (k - ((k >> 1) << 1)) == 0 ? texel.xy : texel.zw;
}
#endif

// Uniforms — batch-wide only.
#ifdef USE_RTE
    uniform vec3 uEyeRTEHigh;
    uniform vec3 uEyeRTELow;
    // Always 1.0 — blocks fast-math reassociation of the high/low
    // recombination (see chunks/rte_pars_vertex.glsl).
    uniform float u_rteOne;
#else
    uniform vec3 uRTCCenter;
    // RTC center already transformed into view (eye) space on the CPU in
    // float64, to avoid the catastrophic float32 cancellation that
    // `viewMatrix * uRTCCenter` suffers from when both operands are ~6.4e6.
    uniform vec3 uRTCCenterView;
#endif

// Current atlas dimensions in pixels. Both update at runtime when the atlas
// grows on overflow, so per-instance pixel rects normalize to the right UV
// regardless of when the geometry was built.
uniform vec2 uSdfAtlasSize;
uniform vec2 uColorAtlasSize;
uniform bool uSizeInMeters;
// Material-level orientation defaults. Per-feature overrides arrive through
// the batch data texture (USE_BATCH_ORIENTATION / USE_BATCH_ROTATION), so
// these are what a label uses until something writes its own value.
uniform bool uFlatFacing;
uniform bool uRotateWithCamera;
// Radians, clockwise seen from the front; converted from the material's
// degrees on the CPU.
uniform float uRotation;
uniform float uFovRad;
uniform float uScreenHeightPx;
uniform vec2 uCenter;
uniform bool uShowBackground;

// Varyings
varying vec2 vAtlasUv;
// Bounds of the current glyph's atlas rectangle. Small-text supersampling in
// the fragment shader clamps taps to these bounds so they cannot read a
// neighboring packed glyph.
flat varying vec2 vAtlasUvMin;
flat varying vec2 vAtlasUvMax;
varying float vFragDepth;
flat varying int vBackGroundSprite; // Whether this vertex belongs to the background sprite (1) or a glyph (0)
flat varying float vBackGroundRatio;
flat varying int vIsColor; // Per-instance flag: glyph is sampled from the color atlas
// Per-label style, resolved here so the fragment shader stays uniform-driven.
flat varying vec3 vColor;
// Style opacity already scaled by the declutter fade.
flat varying float vOpacity;
flat varying float vBatchID;

void main() {
    // Cull unused run slots before any texture reads.
    if (glyphKind == GLYPH_KIND_EMPTY) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // Outside clip space
        return;
    }

    bool isBackground = glyphKind == GLYPH_KIND_BACKGROUND;
    if (isBackground && !uShowBackground) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }

    int slot = int(labelIndex);
    vec4 state = nvr_readLabel(slot, LABEL_ROW_STATE);
    float declutterHide = state.x;
    float show = state.z;

    // Hidden by the evaluator's `show`, or fully faded out by declutter.
    if (show < 0.5 || declutterHide >= 0.999) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    vBatchID = state.y;

    vec4 posHigh = nvr_readLabel(slot, LABEL_ROW_POSITION_HIGH);
    vec4 posLow = nvr_readLabel(slot, LABEL_ROW_POSITION_LOW);
    vec4 box = nvr_readLabel(slot, LABEL_ROW_BOX);

    // Per-feature style from the shared batch data texture (see
    // guide/BATCH_TEXTURE.md), keyed by the feature index in STATE.w. Every
    // existing label has its style written through, so the defaults below
    // only cover the pre-first-write program.
    float nvr_labelBatchIndex = state.w;
    float addHeight = 0.0;
    float batchSize = -1.0;
    float nvr_vShow = 1.0;
    float nvr_vOpacity = 1.0;
    float nvr_batchRotation = uRotation;
    bool nvr_batchFlatFacing = uFlatFacing;
    bool nvr_batchRotateWithCamera = uRotateWithCamera;
    vColor = vec3(1.0);
    #include "chunks/batch_texture_vertex.glsl"

    float fontSize = max(batchSize, 0.0);
    float textWidth = box.x;
    float textHeight = box.y;
    vec2 bgYBounds = box.zw;

    vOpacity = nvr_vOpacity * (1.0 - declutterHide);

#ifdef USE_RTE
    vec3 positionHigh = posHigh.xyz;
    vec3 positionLow = posLow.xyz;
    vec3 absTransformed = positionHigh + positionLow;
#else
    vec3 rtcPosition = posHigh.xyz;
    vec3 absTransformed = rtcPosition + uRTCCenter;
#endif
    #include "chunks/horizon_culling_vertex.glsl"

    vec4 mvPosition;
#ifdef USE_RTE
    // The u_rteOne (== 1.0) factor is load-bearing: see chunks/rte_pars_vertex.glsl.
    vec3 highDiff = (positionHigh - uEyeRTEHigh) * u_rteOne;
    vec3 lowDiff = positionLow - uEyeRTELow;
    vec3 resolvedPosition = highDiff + lowDiff;

    mat4 viewMatrixRTE = viewMatrix;
    viewMatrixRTE[3] = vec4(0.0, 0.0, 0.0, 1.0); // Remove translation
    mvPosition = viewMatrixRTE * vec4(resolvedPosition, 1.0);
#else
    // Adjust view matrix for RTC. uRTCCenterView is the RTC center already in
    // view space (computed on the CPU in float64), so no large-coordinate
    // float32 subtraction happens here — this is what removes the jitter.
    mat4 viewMatrixRTC = viewMatrix;
    viewMatrixRTC[3] = vec4(uRTCCenterView, 1.0);

    mvPosition = viewMatrixRTC * vec4(rtcPosition, 1.0);
#endif

    mvPosition += mvr_getMvHeightOffset(absTransformed, addHeight);

    // Compute scale factor: when sizeInMeters is off, convert pixel size to
    // world units so text maintains constant screen-pixel size at any distance.
    // Normalized text height is 1.0, so no fontSizeWorld division is needed.
    float scaleFactor = fontSize;
    if (!uSizeInMeters) {
        scaleFactor = nvr_pxToWorld(fontSize, uFovRad, uScreenHeightPx, vec3(0.0, 0.0, mvPosition.z), vec3(0.0, 0.0, 0.0));
    }

    vec2 center = clamp(uCenter, vec2(-0.5), vec2(0.5)); // Ensure center is within the bounds of the sprite

    vec3 axisRight;
    vec3 axisUp;
#ifdef NVR_LINE_PLACEMENT
    // A curved label's background would have to be a bent ribbon, which one
    // quad cannot express, so line placement draws glyphs only.
    if (isBackground) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }

    vec4 pathRow = nvr_readLabel(slot, LABEL_ROW_PATH);
    // The placement pass rejected this label: it is longer than the road it
    // sits on, or the road bends too sharply under it to stay readable. Culled
    // rather than faded, because unlike a declutter loss this is not a
    // competition the label could win back by a pixel of camera drift.
    if (pathRow.w > 0.5) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }

    int pathBase = int(pathRow.x);
    float stepMeters = pathRow.y;
    // keepUpright walks the path backwards, so a label never reads
    // right-to-left. Reversing the tangent with it keeps the glyph frame
    // self-consistent: the text's own "up" stays on its own up side.
    float dir = pathRow.z > 0.5 ? -1.0 : 1.0;

    // Follow the line a WORD at a time. The word is placed rigidly at its own
    // centre and its glyphs are laid along that one tangent, so the letters of
    // a name stay square to each other however the road bends under them —
    // per-glyph tangents splay them apart on a tight curve and the word stops
    // reading as a word. Only the joins between words take up the curvature.
    //
    // Quads are never bent vertex by vertex either way: that would shear them.
    float wordCenterEm = glyphWordCenter - center.x * textWidth;
    float sMeters = wordCenterEm * scaleFactor * dir;

    float halfSpan = 0.5 * float(PATH_SAMPLES - 1) * stepMeters;
    float t = (sMeters + halfSpan) / max(stepMeters, 1e-6);
    int seg = int(clamp(floor(t), 0.0, float(PATH_SAMPLES - 2)));
    vec2 pa = nvr_readPath(pathBase, seg);
    vec2 pb = nvr_readPath(pathBase, seg + 1);
    vec2 pathPos = mix(pa, pb, clamp(t - float(seg), 0.0, 1.0));

    vec2 tangent = pb - pa;
    float tangentLen = length(tangent);
    // A doubled-back hairpin can put two samples on the same point; east keeps
    // the glyph readable rather than letting a normalize() produce NaN.
    tangent = (tangentLen > 1e-6 ? tangent / tangentLen : vec2(1.0, 0.0)) * dir;

    // Samples are evenly spaced in ARC length but joined by straight chords, so
    // a segment containing a bend covers less ground than the arc it stands
    // for, and `mix` advances more slowly than the label's own em ruler.
    //
    // Placing every glyph individually hid this: all of them shrank by the same
    // factor, so the label just came out slightly short. A rigid word does not
    // shrink, so the whole of the discrepancy is taken out of the gaps between
    // words instead — which is where the spaces went. Put the glyphs on the
    // path's ruler too, and words and gaps shrink together again.
    //
    // Chord never exceeds arc, so the clamp only guards f32 noise.
    float pathScale = min(tangentLen / max(stepMeters, 1e-6), 1.0);
    vec2 normal = vec2(-tangent.y, tangent.x);

    // `scaleFactor` is metres per em, so dividing by the font size recovers
    // metres per style unit — which is what lineOffset is expressed in.
    pathPos += normal * (uLineOffset * (scaleFactor / max(fontSize, 1e-6)));

    vec3 eastWorld, northWorld, normalWorld;
    nvr_enuBasis(absTransformed, eastWorld, northWorld, normalWorld);
    vec3 eastView = (viewMatrix * vec4(eastWorld, 0.0)).xyz;
    vec3 northView = (viewMatrix * vec4(northWorld, 0.0)).xyz;

    // The walk decides where in the tangent plane the glyph sits; facing still
    // decides which plane its quad stands in.
    axisRight = tangent.x * eastView + tangent.y * northView;
    axisUp = nvr_batchFlatFacing
        ? normal.x * eastView + normal.y * northView
        : (viewMatrix * vec4(normalWorld, 0.0)).xyz;

    // Path offsets are already metres, so they bypass the em scaling below.
    mvPosition.xyz += pathPos.x * eastView + pathPos.y * northView;
#else
    nvr_quadBasis(
        absTransformed,
        nvr_batchFlatFacing,
        nvr_batchRotateWithCamera,
        nvr_batchRotation,
        axisRight,
        axisUp
    );
#endif

    vIsColor = glyphKind == GLYPH_KIND_COLOR ? 1 : 0;

    if (isBackground) {
        vBackGroundSprite = 1;

        float bgHeight = bgYBounds.y - bgYBounds.x;
        vec2 bgLocalPos = (position.xy + vec2(0.5)) * vec2(textWidth, bgHeight) + vec2(0.0, bgYBounds.x);
        bgLocalPos.x -= center.x * textWidth;
        bgLocalPos.y -= center.y * textHeight;

        vec4 newMvPosition = mvPosition
            + vec4((bgLocalPos.x * axisRight + bgLocalPos.y * axisUp) * scaleFactor, 0.0);

        gl_Position = projectionMatrix * newMvPosition;

        vAtlasUv = uv;
        vBackGroundRatio = textWidth / bgHeight; // Pass the aspect ratio of the background sprite to the fragment shader for proper corner radius scaling
    } else {
        vBackGroundSprite = 0;
        // --- Per-glyph vertex position ---
        // position.xy is the unit quad [-0.5, 0.5].
        // glyphOffset is the glyph bbox min corner (left/bottom), so remap
        // the centered quad to [0,1] before applying glyph size/offset.
        vec2 localPos = (position.xy + vec2(0.5)) * glyphSize + glyphOffset;

        // Apply centering: shift entire text block by anchor point
        localPos.x -= center.x * textWidth;
        localPos.y -= center.y * textHeight;

#ifdef NVR_LINE_PLACEMENT
        // The walk placed this glyph's WORD on the curve, so what is left is
        // the offset from the word's centre, laid along the word's single
        // tangent. Only x needs rebasing: y is still the baseline-relative
        // height the layout gave it.
        //
        // Split in two so `pathScale` reaches the spacing but not the
        // letterforms: where the glyph sits inside its word rides the path's
        // ruler, while the quad's own corners keep the glyph's true shape.
        float glyphCenterEm = glyphOffset.x + glyphSize.x * 0.5 - center.x * textWidth;
        localPos.x =
            (glyphCenterEm - wordCenterEm) * pathScale + (localPos.x - glyphCenterEm);
#endif

        // Lay the glyph out in the label's basis (see nvr_quadBasis), scaled.
        vec4 delta = vec4((localPos.x * axisRight + localPos.y * axisUp) * scaleFactor, 0.0);
        vec4 newMvPosition = mvPosition + delta;

        gl_Position = projectionMatrix * newMvPosition;

        // Atlas UV interpolation: glyphUvRect carries pixel-space corners so
        // resizing the atlas only requires updating the size uniform — geometry
        // attributes stay valid.
        vec2 atlasSize = vIsColor == 1 ? uColorAtlasSize : uSdfAtlasSize;
        vAtlasUvMin = glyphUvRect.xy / atlasSize;
        vAtlasUvMax = glyphUvRect.zw / atlasSize;
        vAtlasUv = mix(vAtlasUvMin, vAtlasUvMax, uv);
    }

    vFragDepth = gl_Position.w + 1.0;
}
