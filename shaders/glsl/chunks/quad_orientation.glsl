#ifndef QUAD_ORIENTATION_GLSL
#define QUAD_ORIENTATION_GLSL

/**
 * View-space basis an anchored quad (a text label, a sprite) is laid out in:
 * `right` spans the quad's local +x, `up` its local +y. A vertex is then
 * `mvPosition + vec4(localPos.x * right + localPos.y * up, 0.0)`.
 *
 * A view-space offset with zero Z is what makes a quad screen-aligned, so the
 * classic billboard is just the view-space x/y axes; the other modes swap in a
 * pair derived from the anchor's local east-north-up frame. Shared by
 * sdfText.vert.glsl and instancedSprite.vert.glsl so the two cannot drift.
 *
 * `flatFacing`  false = the quad stands up, true = it lies in the ellipsoid's
 *               tangent plane at the anchor.
 * `rotateWithCamera` true = the quad turns to follow the camera, false = it is
 *               frozen in the anchor's east-north-up frame.
 *
 * `worldPos` is the anchor in ECEF meters; its normalized direction is the
 * surface normal, matching mvr_getMvHeightOffset's spherical approximation.
 */
/**
 * The anchor's local east-north-up frame, in WORLD space.
 *
 * `cross(polar axis, normal)` vanishes at the poles, where every tangent
 * direction is an equally valid "east"; ECEF +x is the fallback there.
 *
 * The surface normal is the normalized anchor direction — the same spherical
 * approximation mvr_getMvHeightOffset uses, so height offsets and quad bases
 * cannot disagree about which way is up.
 */
void nvr_enuBasis(vec3 worldPos, out vec3 east, out vec3 north, out vec3 up) {
    vec3 nWorld = normalize(worldPos);
    vec3 e = vec3(-nWorld.y, nWorld.x, 0.0);
    float eLen = length(e);
    east = eLen > 1e-6 ? e / eLen : vec3(1.0, 0.0, 0.0);
    north = cross(nWorld, east);
    up = nWorld;
}

void nvr_quadOrientation(
    vec3 worldPos,
    bool flatFacing,
    bool rotateWithCamera,
    out vec3 right,
    out vec3 up
) {
    if (!flatFacing && rotateWithCamera) {
        // Screen plane, screen up — the default billboard.
        right = vec3(1.0, 0.0, 0.0);
        up = vec3(0.0, 1.0, 0.0);
        return;
    }

    vec3 nWorld = normalize(worldPos);

    if (flatFacing && rotateWithCamera) {
        // Tangent plane, yawed so the text still reads left-to-right: screen
        // right projected onto the plane. That projection degenerates only
        // when the normal points along screen x, where the quad is edge-on
        // anyway.
        vec3 n = (viewMatrix * vec4(nWorld, 0.0)).xyz;
        vec3 t = vec3(1.0, 0.0, 0.0) - n.x * n;
        float tLen = length(t);
        right = tLen > 1e-4 ? t / tLen : normalize(vec3(0.0, 1.0, 0.0) - n.y * n);
        up = cross(n, right);
        return;
    }

    // Both no-rotate modes are pinned to the anchor's east-north-up frame, so
    // the basis depends only on the anchor and the camera cannot disturb it.
    vec3 eastWorld, northWorld, normalWorld;
    nvr_enuBasis(worldPos, eastWorld, northWorld, normalWorld);
    // Flat lies in the tangent plane with up = north. Upright stands the quad
    // on the surface with up = the surface normal, a signboard whose face
    // points south — readable from a camera looking northward.
    vec3 upWorld = flatFacing ? northWorld : normalWorld;
    right = (viewMatrix * vec4(eastWorld, 0.0)).xyz;
    up = (viewMatrix * vec4(upWorld, 0.0)).xyz;
}

/**
 * The basis above, then spun by `rotation` radians inside the quad's own
 * plane. Both axes turn together, so the quad rotates rigidly and never leaves
 * the plane its orientation mode put it in.
 *
 * Local offsets are measured from the anchor, so this pivots the quad about
 * that anchor — which makes the material's `center` the control for where the
 * pivot sits inside the quad.
 */
void nvr_quadBasis(
    vec3 worldPos,
    bool flatFacing,
    bool rotateWithCamera,
    float rotation,
    out vec3 right,
    out vec3 up
) {
    nvr_quadOrientation(worldPos, flatFacing, rotateWithCamera, right, up);
    if (rotation == 0.0) {
        return;
    }
    // Negated sine relative to the usual counter-clockwise matrix, so a
    // positive angle reads clockwise from in front of the quad.
    float s = sin(rotation);
    float c = cos(rotation);
    vec3 rotatedRight = c * right - s * up;
    up = s * right + c * up;
    right = rotatedRight;
}

#endif // QUAD_ORIENTATION_GLSL
