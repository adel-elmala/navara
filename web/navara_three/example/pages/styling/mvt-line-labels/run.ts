import ThreeView, { Color, type Layer } from "@navaramap/three";
import { DefaultPlugin } from "@navaramap/three-default-plugin";
import { Pane } from "tweakpane";

import {
  FONT_DATASETS,
  TILE_DATASETS,
  VECTOR_DATASETS,
} from "../../../helpers/constants";

/**
 * Street-name labels repeated along road centerlines, bending with the road.
 *
 * The data is OpenMapTiles-schema: `transportation` carries road geometry and
 * `transportation_name` carries `name`/`class` on linestrings merged by name.
 * Labelling the merged layer is what keeps one name per road rather than one
 * per segment — though plenty of those merged lines are still only tens of
 * metres long, and a label that would overrun its road is dropped.
 */

/** Road classes worth drawing, widest first. Everything else is filtered out. */
const ROAD_WIDTH: Record<string, number> = {
  motorway: 10,
  trunk: 8,
  primary: 7,
  secondary: 5,
  tertiary: 4,
  minor: 3,
  service: 2,
  residential: 3,
};

/** Bigger roads win an overlap in the declutter pass. */
const LABEL_PRIORITY: Record<string, number> = {
  motorway: 5,
  trunk: 4,
  primary: 3,
  secondary: 2,
  tertiary: 1,
};

const params = {
  placement: "line" as "point" | "line" | "line-center",
  spacing: 250,
  maxAngle: 45,
  keepUpright: true,
  lineOffset: 0,
  size: 9,
  sizeInMeters: false,
  outlineWidth: 2,
  declutter: true,
};

export const run = async (view: ThreeView) => {
  const defaultPlugin = new DefaultPlugin();
  view.addPlugin(defaultPlugin);

  await view.init();

  // Central London: dense named streets that actually curve, which is what
  // `maxAngle` and the curved layout are there to handle. Close enough that a
  // street is many times longer on screen than its name:
  // a label is dropped when it would overrun the road it sits on, so pulling
  // back thins the labels out exactly as it does on any other map.
  view.setCamera({
    lng: -0.1276,
    lat: 51.5105,
    height: 1500,
    heading: 0,
    pitch: -55,
    roll: 0,
  });

  // Satellite imagery rather than a street map: draped geometry needs a surface
  // to composite onto, and a basemap with its own baked-in street names would
  // make it impossible to tell which labels this layer drew.
  const basemap = view.addSource({
    type: "raster-tile",
    url: TILE_DATASETS.eox.url,
    maxZoom: 16,
  });
  view.addLayer({ type: "raster", source: basemap });

  const planet = view.addSource({
    type: "vector-tile",
    url: VECTOR_DATASETS.openFreeMapPlanet.url,
    maxZoom: 14,
  });

  // Road geometry, for the labels to sit on.
  const roads = view.addLayer({
    type: "vector",
    source: planet,
    sourceLayers: ["transportation"],
    polyline: {
      color: new Color().setStyle("#ffb454"),
      width: 4,
      clampToGround: true,
      geometryTypes: ["line"],
    },
  });

  roads.on("featureUpdated", ({ evaluator }) => {
    evaluator.evaluate(
      ({ properties }) => {
        const width = ROAD_WIDTH[properties?.["class"] as string];
        return width === undefined ? { show: false } : { width };
      },
      { filters: ["class"] },
    );
  });

  // The labels themselves. `geometryTypes: ["line"]` opts the text appearance
  // into line geometry; `placement` then decides whether that means one label
  // per vertex (the historical behaviour) or labels spaced along the line.
  const labels = view.addLayer({
    type: "vector",
    source: planet,
    sourceLayers: ["transportation_name"],
    text: {
      font: FONT_DATASETS.Roboto.url,
      geometryTypes: ["line"],
      placement: params.placement,
      spacing: params.spacing,
      maxAngle: params.maxAngle,
      keepUpright: params.keepUpright,
      lineOffset: params.lineOffset,
      // Line placement always lays the label in the ground plane and takes its
      // direction from the line, so `rotateWithCamera` has no effect here.
      textFacing: "flat",
      size: params.size,
      sizeInMeters: params.sizeInMeters,
      clampToGround: true,
      color: new Color().setStyle("#ffffff"),
      outlineColor: new Color().setStyle("#111318"),
      outlineWidth: params.outlineWidth,
      declutter: params.declutter,
    },
  });

  labels.on("featureUpdated", ({ evaluator }) => {
    evaluator.evaluate(
      ({ properties }) => {
        const name = properties?.["name"] as string | undefined;
        if (!name) return { show: false, text: "" };
        return {
          text: name,
          show: true,
          declutterPriority:
            LABEL_PRIORITY[properties?.["class"] as string] ?? 0,
        };
      },
      { filters: ["name", "class"] },
    );
  });

  addControls(view, labels);

  view.attribution?.add([
    TILE_DATASETS.eox,
    VECTOR_DATASETS.openFreeMapPlanet,
    FONT_DATASETS.Roboto,
  ]);
};

const addControls = (view: ThreeView, labels: Layer) => {
  const pane = new Pane({ title: "Line Labels" });

  const update = () => {
    labels.update({
      text: {
        placement: params.placement,
        spacing: params.spacing,
        maxAngle: params.maxAngle,
        keepUpright: params.keepUpright,
        lineOffset: params.lineOffset,
        size: params.size,
        sizeInMeters: params.sizeInMeters,
        outlineWidth: params.outlineWidth,
        declutter: params.declutter,
      },
    });
    view.forceUpdate();
  };

  const placement = pane.addFolder({ title: "Placement" });
  placement
    .addBinding(params, "placement", {
      options: {
        "along the line": "line",
        "line midpoint": "line-center",
        "per vertex": "point",
      },
    })
    .on("change", update);
  // Spacing and maxAngle are resolved when a tile is parsed, so changing them
  // only affects tiles fetched afterwards.
  placement
    .addBinding(params, "spacing", { min: 60, max: 800, step: 10 })
    .on("change", update);
  placement
    .addBinding(params, "maxAngle", { min: 5, max: 180, step: 5 })
    .on("change", update);
  placement.addBinding(params, "keepUpright").on("change", update);
  placement
    .addBinding(params, "lineOffset", { min: -30, max: 30, step: 1 })
    .on("change", update);

  const style = pane.addFolder({ title: "Style" });
  style
    .addBinding(params, "size", { min: 6, max: 48, step: 1 })
    .on("change", update);
  style.addBinding(params, "sizeInMeters").on("change", update);
  style
    .addBinding(params, "outlineWidth", { min: 0, max: 6, step: 0.5 })
    .on("change", update);
  style.addBinding(params, "declutter").on("change", update);
};
