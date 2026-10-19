#!/usr/bin/env node
/*
 * Renders a deck spec (JSON written by the project's Python code) into a PPTX.
 *
 *   node deck/build_deck.js deck/deck_spec.json deck/<name>.pptx
 *
 * The spec carries every word and number on the slides; this file only lays
 * them out. Image paths in the spec are relative to the spec file.
 *
 * Slide types: title, finding, chart, two_charts, stats, steps, cards,
 * columns, table, text. See STANDARDS.md for the fields and length limits.
 */
"use strict";

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const W = 13.333;
const H = 7.5;
const M = 0.6; // outer margin
const CW = W - 2 * M; // content width
const TOP = 1.95; // body starts here
const BOTTOM = 6.75; // body ends here
const FONT = "Arial";

const C = {
  ink: "1A1A1A",
  ink2: "52514E",
  muted: "6F6E69",
  line: "D9D7CF",
  white: "FFFFFF",
};

// ---------------------------------------------------------------- helpers

function pngSize(file) {
  const b = fs.readFileSync(file);
  if (b.readUInt32BE(0) !== 0x89504e47) throw new Error(`Not a PNG: ${file}`);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

// Fit an image inside a box without distortion, anchored top-left.
function fitTopLeft(file, box) {
  const { w, h } = pngSize(file);
  const r = Math.min(box.w / w, box.h / h);
  return { x: box.x, y: box.y, w: w * r, h: h * r };
}

function needs(spec, keys, idx) {
  for (const k of keys) {
    if (spec[k] === undefined || spec[k] === null || spec[k] === "") {
      throw new Error(`Slide ${idx + 1} (${spec.type}): missing "${k}"`);
    }
  }
}

function bulletRuns(items, opts = {}) {
  return items.map((t, i) => ({
    text: t,
    options: {
      bullet: { indent: 14 },
      breakLine: i < items.length - 1,
      paraSpaceAfter: opts.space ?? 8,
    },
  }));
}

function text(slide, value, o) {
  slide.addText(value, { fontFace: FONT, margin: 0, isTextBox: true, valign: "top", ...o });
}

function kicker(slide, t, value, dark = false) {
  if (!value) return;
  text(slide, value.toUpperCase(), {
    x: M, y: 0.38, w: CW, h: 0.3,
    fontSize: 11, bold: true, color: t.accent, charSpacing: 1.5,
    objectName: "Kicker",
  });
}

function title(slide, value) {
  slide.addText(value, { placeholder: "title" });
}

function panel(slide, t, box, name = "Panel") {
  slide.addShape("roundRect", {
    ...box, fill: { color: t.panel }, line: { color: t.panel, width: 0 },
    rectRadius: 0.08, objectName: name,
  });
}

function addImage(slide, base, rel, box) {
  const file = path.resolve(base, rel);
  if (!fs.existsSync(file)) throw new Error(`Image not found: ${file}`);
  slide.addImage({ path: file, ...fitTopLeft(file, box), objectName: path.basename(rel) });
}

function footnote(slide, value, y = 6.42) {
  if (!value) return;
  text(slide, value, { x: M, y, w: CW, h: 0.4, fontSize: 10, color: C.muted, valign: "bottom", objectName: "Footnote" });
}

// ---------------------------------------------------------------- layouts

function defineLayouts(pres, t) {
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: C.white },
    margin: [0.5, M, 0.5, M],
    objects: [
      {
        placeholder: {
          options: {
            name: "title", type: "title", x: M, y: 0.72, w: CW, h: 1.05,
            fontFace: FONT, fontSize: 26, bold: true, color: C.ink, valign: "top", align: "left", margin: 0,
          },
          text: "",
        },
      },
      {
        text: {
          text: t.footer,
          options: { x: M, y: 7.0, w: 10, h: 0.3, fontFace: FONT, fontSize: 10, color: C.muted, margin: 0, valign: "middle" },
        },
      },
    ],
    slideNumber: { x: W - M - 0.8, y: 7.0, w: 0.8, h: 0.3, fontFace: FONT, fontSize: 10, color: C.muted, align: "right" },
  });

  pres.defineSlideMaster({
    title: "COVER",
    background: { color: t.dark },
    objects: [
      {
        placeholder: {
          options: {
            name: "title", type: "title", x: M, y: 2.05, w: 11.2, h: 1.6,
            fontFace: FONT, fontSize: 44, bold: true, color: C.white, valign: "top", align: "left", margin: 0,
          },
          text: "",
        },
      },
    ],
  });
}

// ---------------------------------------------------------------- slides

const SLIDES = {
  title(pres, t, s, base, i) {
    needs(s, ["title", "subtitle", "disclaimer", "author"], i);
    const slide = pres.addSlide({ masterName: "COVER", sectionTitle: s.section });
    if (s.kicker) {
      text(slide, s.kicker.toUpperCase(), { x: M, y: 1.55, w: 11, h: 0.35, fontSize: 13, bold: true, color: t.accent, charSpacing: 2 });
    }
    title(slide, s.title);
    text(slide, s.subtitle, { x: M, y: 3.75, w: 10.8, h: 1.1, fontSize: 20, color: t.coverText });
    // Fictional-case-study notice: always on the cover, never optional.
    slide.addShape("roundRect", {
      x: M, y: 5.15, w: 8.6, h: 0.62, fill: { color: t.coverPanel }, line: { color: t.coverPanel, width: 0 },
      rectRadius: 0.1, objectName: "Disclaimer panel",
    });
    text(slide, s.disclaimer, { x: M + 0.25, y: 5.15, w: 8.2, h: 0.62, fontSize: 12.5, color: C.white, valign: "middle" });
    text(slide, s.author, { x: M, y: 6.45, w: 11, h: 0.4, fontSize: 13, color: t.coverMuted });
    if (s.notes) slide.addNotes(s.notes);
  },

  // Chart left, commentary panel right: big stat + up to 3 bullets.
  finding(pres, t, s, base, i) {
    needs(s, ["title", "image"], i);
    const slide = pres.addSlide({ masterName: "CONTENT", sectionTitle: s.section });
    kicker(slide, t, s.kicker);
    title(slide, s.title);
    addImage(slide, base, s.image, { x: M, y: TOP, w: 8.1, h: 4.8 });
    const px = 9.0, pw = W - M - px;
    panel(slide, t, { x: px, y: TOP, w: pw, h: 4.8 });
    let y = TOP + 0.3;
    if (s.stat) {
      text(slide, s.stat.value, { x: px + 0.3, y, w: pw - 0.6, h: 0.85, fontSize: 40, bold: true, color: t.accent, fit: "none" });
      y += 0.88;
      text(slide, s.stat.label, { x: px + 0.3, y, w: pw - 0.6, h: 0.75, fontSize: 13, color: C.ink2 });
      y += 0.95;
    }
    if (s.bullets && s.bullets.length) {
      text(slide, bulletRuns(s.bullets), { x: px + 0.3, y, w: pw - 0.6, h: TOP + 4.8 - y - 0.45, fontSize: 14, color: C.ink });
    }
    if (s.source) {
      text(slide, s.source, { x: px + 0.3, y: TOP + 4.8 - 0.45, w: pw - 0.6, h: 0.32, fontSize: 9.5, color: C.muted, valign: "bottom" });
    }
    if (s.notes) slide.addNotes(s.notes);
  },

  chart(pres, t, s, base, i) {
    needs(s, ["title", "image"], i);
    const slide = pres.addSlide({ masterName: "CONTENT", sectionTitle: s.section });
    kicker(slide, t, s.kicker);
    title(slide, s.title);
    addImage(slide, base, s.image, { x: M, y: TOP, w: CW, h: 4.5 });
    footnote(slide, s.caption);
    if (s.notes) slide.addNotes(s.notes);
  },

  two_charts(pres, t, s, base, i) {
    needs(s, ["title", "images"], i);
    const slide = pres.addSlide({ masterName: "CONTENT", sectionTitle: s.section });
    kicker(slide, t, s.kicker);
    title(slide, s.title);
    const gap = 0.4, w = (CW - gap) / 2;
    s.images.forEach((im, k) => {
      const x = M + k * (w + gap);
      addImage(slide, base, im.image, { x, y: TOP, w, h: 4.2 });
      if (im.caption) text(slide, im.caption, { x, y: TOP + 4.3, w, h: 0.45, fontSize: 11, color: C.ink2 });
    });
    if (s.notes) slide.addNotes(s.notes);
  },

  // 2-4 large numbers, optional lead sentence above.
  stats(pres, t, s, base, i) {
    needs(s, ["title", "stats"], i);
    const slide = pres.addSlide({ masterName: "CONTENT", sectionTitle: s.section });
    kicker(slide, t, s.kicker);
    title(slide, s.title);
    let y = TOP;
    if (s.lead) {
      text(slide, s.lead, { x: M, y, w: CW, h: 0.85, fontSize: 16, color: C.ink2 });
      y += 1.05;
    }
    const n = s.stats.length, gap = 0.3, w = (CW - gap * (n - 1)) / n, h = Math.min(2.9, 6.25 - y);
    s.stats.forEach((st, k) => {
      const x = M + k * (w + gap);
      panel(slide, t, { x, y, w, h }, `Stat ${k + 1}`);
      text(slide, st.value, { x: x + 0.3, y: y + 0.3, w: w - 0.6, h: 0.95, fontSize: n > 3 ? 36 : 44, bold: true, color: t.accent, fit: "none" });
      text(slide, st.label, { x: x + 0.3, y: y + 1.3, w: w - 0.6, h: 0.75, fontSize: 15, bold: true, color: C.ink });
      if (st.note) text(slide, st.note, { x: x + 0.3, y: y + 2.0, w: w - 0.6, h: h - 2.1, fontSize: 12, color: C.ink2 });
    });
    footnote(slide, s.footnote);
    if (s.notes) slide.addNotes(s.notes);
  },

  // Numbered process, 3-5 steps.
  steps(pres, t, s, base, i) {
    needs(s, ["title", "steps"], i);
    const slide = pres.addSlide({ masterName: "CONTENT", sectionTitle: s.section });
    kicker(slide, t, s.kicker);
    title(slide, s.title);
    const n = s.steps.length, gap = 0.35, w = (CW - gap * (n - 1)) / n, d = 0.62, y = TOP + 0.15;
    for (let k = 0; k < n - 1; k++) {
      const x1 = M + k * (w + gap) + d + 0.12, x2 = M + (k + 1) * (w + gap) - 0.12;
      slide.addShape("line", { x: x1, y: y + d / 2, w: x2 - x1, h: 0, line: { color: C.line, width: 1.25 }, objectName: `Connector ${k + 1}` });
    }
    s.steps.forEach((st, k) => {
      const x = M + k * (w + gap);
      slide.addShape("ellipse", { x, y, w: d, h: d, fill: { color: t.accent }, line: { color: t.accent, width: 0 }, objectName: `Step ${k + 1} marker` });
      text(slide, String(k + 1), { x, y, w: d, h: d, fontSize: 18, bold: true, color: C.white, align: "center", valign: "middle" });
      text(slide, st.title, { x, y: y + d + 0.3, w, h: 0.45, fontSize: 17, bold: true, color: C.ink });
      text(slide, st.text, { x, y: y + d + 0.85, w, h: 6.3 - (y + d + 0.85), fontSize: 14, color: C.ink2 });
    });
    footnote(slide, s.footnote);
    if (s.notes) slide.addNotes(s.notes);
  },

  // Recommendation cards: tag pill, title, text, meta line.
  cards(pres, t, s, base, i) {
    needs(s, ["title", "cards"], i);
    const slide = pres.addSlide({ masterName: "CONTENT", sectionTitle: s.section });
    kicker(slide, t, s.kicker);
    title(slide, s.title);
    const n = s.cards.length, gap = 0.3, w = (CW - gap * (n - 1)) / n, y = TOP, h = s.footnote ? 4.35 : 4.75;
    s.cards.forEach((c, k) => {
      const x = M + k * (w + gap);
      panel(slide, t, { x, y, w, h }, `Card ${k + 1}`);
      if (c.tag) {
        const tw = Math.min(w - 0.6, 0.18 + c.tag.length * 0.1);
        slide.addShape("roundRect", { x: x + 0.3, y: y + 0.3, w: tw, h: 0.34, fill: { color: t.accentSoft }, line: { color: t.accentSoft, width: 0 }, rectRadius: 0.17, objectName: `Card ${k + 1} tag` });
        text(slide, c.tag.toUpperCase(), { x: x + 0.3, y: y + 0.3, w: tw, h: 0.34, fontSize: 10, bold: true, color: t.accentText, align: "center", valign: "middle", charSpacing: 1 });
      }
      text(slide, c.title, { x: x + 0.3, y: y + 0.85, w: w - 0.6, h: 0.95, fontSize: 17, bold: true, color: C.ink });
      text(slide, c.text, { x: x + 0.3, y: y + 1.85, w: w - 0.6, h: h - 2.75, fontSize: 14, color: C.ink2 });
      if (c.meta) text(slide, c.meta, { x: x + 0.3, y: y + h - 0.8, w: w - 0.6, h: 0.55, fontSize: 12, bold: true, color: C.ink, valign: "bottom" });
    });
    footnote(slide, s.footnote);
    if (s.notes) slide.addNotes(s.notes);
  },

  // 2-3 columns of bullets on tinted panels (data overview, limitations / next steps).
  columns(pres, t, s, base, i) {
    needs(s, ["title", "columns"], i);
    const slide = pres.addSlide({ masterName: "CONTENT", sectionTitle: s.section });
    kicker(slide, t, s.kicker);
    title(slide, s.title);
    const n = s.columns.length, gap = 0.3, w = (CW - gap * (n - 1)) / n, y = TOP, h = s.footnote ? 4.35 : 4.75;
    s.columns.forEach((col, k) => {
      const x = M + k * (w + gap);
      panel(slide, t, { x, y, w, h }, `Column ${k + 1}`);
      text(slide, col.heading, { x: x + 0.3, y: y + 0.3, w: w - 0.6, h: 0.45, fontSize: 17, bold: true, color: C.ink });
      text(slide, bulletRuns(col.bullets, { space: 10 }), { x: x + 0.3, y: y + 0.95, w: w - 0.6, h: h - 1.2, fontSize: 14, color: C.ink2 });
    });
    footnote(slide, s.footnote);
    if (s.notes) slide.addNotes(s.notes);
  },

  table(pres, t, s, base, i) {
    needs(s, ["title", "header", "rows"], i);
    const slide = pres.addSlide({ masterName: "CONTENT", sectionTitle: s.section });
    kicker(slide, t, s.kicker);
    title(slide, s.title);
    const border = { type: "solid", pt: 0.75, color: C.line };
    const head = s.header.map((h) => ({ text: h, options: { bold: true, color: C.ink, fill: { color: t.panel } } }));
    const rows = s.rows.map((r) => r.map((c) => ({ text: String(c), options: { color: C.ink2 } })));
    slide.addTable([head, ...rows], {
      x: M, y: TOP, w: CW, colW: s.colW, fontFace: FONT, fontSize: s.fontSize || 13,
      border, margin: [5, 8, 5, 8], valign: "middle", autoPage: false,
    });
    footnote(slide, s.footnote);
    if (s.notes) slide.addNotes(s.notes);
  },

  // Appendix / methodology: headed paragraphs in two columns.
  text(pres, t, s, base, i) {
    needs(s, ["title", "sections"], i);
    const slide = pres.addSlide({ masterName: "CONTENT", sectionTitle: s.section });
    kicker(slide, t, s.kicker);
    title(slide, s.title);
    const gap = 0.5, w = (CW - gap) / 2;
    const half = Math.ceil(s.sections.length / 2);
    [s.sections.slice(0, half), s.sections.slice(half)].forEach((group, k) => {
      const runs = [];
      group.forEach((sec, j) => {
        runs.push({ text: sec.heading, options: { bold: true, color: C.ink, fontSize: 13, breakLine: true, paraSpaceAfter: 2 } });
        runs.push({ text: sec.text, options: { color: C.ink2, fontSize: 12, breakLine: j < group.length - 1, paraSpaceAfter: 10 } });
      });
      if (runs.length) text(slide, runs, { x: M + k * (w + gap), y: TOP, w, h: BOTTOM - TOP });
    });
    if (s.notes) slide.addNotes(s.notes);
  },
};

// ---------------------------------------------------------------- main

async function main() {
  const [specPath, outPath] = process.argv.slice(2);
  if (!specPath || !outPath) {
    console.error("usage: node build_deck.js <deck_spec.json> <out.pptx>");
    process.exit(2);
  }
  const spec = JSON.parse(fs.readFileSync(specPath, "utf8"));
  const base = path.dirname(path.resolve(specPath));
  const t = spec.theme;
  for (const k of ["dark", "accent", "accentSoft", "accentText", "panel", "coverText", "coverMuted", "coverPanel", "footer"]) {
    if (!t[k]) throw new Error(`theme.${k} missing`);
  }
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.author = spec.meta.author;
  pres.title = spec.meta.title;
  pres.subject = spec.meta.subject || "";
  pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
  defineLayouts(pres, t);

  let section = null;
  spec.slides.forEach((s, i) => {
    if (s.section && s.section !== section) {
      pres.addSection({ title: s.section });
      section = s.section;
    }
    const fn = SLIDES[s.type];
    if (!fn) throw new Error(`Slide ${i + 1}: unknown type "${s.type}"`);
    fn(pres, t, s, base, i);
  });
  await pres.writeFile({ fileName: outPath });
  console.log(`wrote ${outPath} (${spec.slides.length} slides)`);
}

main().catch((e) => {
  console.error(e.message || e);
  process.exit(1);
});
