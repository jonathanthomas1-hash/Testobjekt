/* ————————————————————————————————————————————————
   Kokon · App-Logik
   Gedanken aufnehmen → im Kokon sammeln → Metamorphose
   (Claude Opus 5) → Meisterwerk. Alles lokal gespeichert.
   ———————————————————————————————————————————————— */

"use strict";

const STORE = {
  gedanken: "kokon.gedanken",
  werk: "kokon.werk",
  schluessel: "kokon.schluessel",
};

const API_URL = "https://api.anthropic.com/v1/messages";
const MODEL = "claude-opus-5";

const WERK_SCHEMA = {
  type: "object",
  properties: {
    titel: { type: "string", description: "Poetischer, prägnanter Titel des Meisterwerks" },
    untertitel: { type: "string", description: "Ein Satz, der das Ganze zusammenfasst" },
    essenz: { type: "string", description: "2–4 Sätze: der rote Faden hinter allen Gedanken" },
    themen: {
      type: "array",
      description: "Die Gedanken zu Themen verdichtet",
      items: {
        type: "object",
        properties: {
          name: { type: "string" },
          symbol: { type: "string", description: "Ein einzelnes passendes Emoji" },
          kern: { type: "string", description: "1–2 Sätze: worum es hier wirklich geht" },
          gedanken: { type: "array", items: { type: "string" }, description: "Die zugehörigen Original-Gedanken, behutsam geglättet" },
          schritte: { type: "array", items: { type: "string" }, description: "2–4 konkrete, entspannte nächste Schritte" },
        },
        required: ["name", "symbol", "kern", "gedanken", "schritte"],
        additionalProperties: false,
      },
    },
    erste_schritte: {
      type: "array",
      items: { type: "string" },
      description: "Die 3 wichtigsten allerersten Schritte über alle Themen hinweg",
    },
    impuls: { type: "string", description: "Ein kurzer, warmer Gedanke zum Mitnehmen – kein Kalenderspruch" },
  },
  required: ["titel", "untertitel", "essenz", "themen", "erste_schritte", "impuls"],
  additionalProperties: false,
};

const SYSTEM_PROMPT = `Du bist die Metamorphose im Herzen von Kokon – einer App, in die Menschen zwischen Yoga, Laufen und Kaffee lose Gedanken werfen, um danach wieder ihr Leben zu leben.

Deine Aufgabe: Verwandle die hineingeworfenen Roh-Gedanken in ein ruhiges, klares Meisterwerk.

Grundsätze:
- Antworte auf Deutsch, warm und klar, ohne Business-Floskeln und ohne Kitsch.
- Erfinde nichts hinzu. Verdichte, ordne und benenne, was wirklich da ist.
- Halbe Sätze und Fragmente ernst nehmen – oft steckt dort die eigentliche Idee.
- Schritte sollen entspannt machbar sein: klein, konkret, ohne Druck.
- Wenn Gedanken sich widersprechen, benenne die Spannung als eigenes Thema statt sie zu glätten.`;

/* ——— Zustand ——— */

let gedanken = load(STORE.gedanken, []);
let werk = load(STORE.werk, null);
let morphing = false;

/* ——— DOM ——— */

const $ = (id) => document.getElementById(id);
const el = {
  viewStudio: $("view-studio"),
  viewWerk: $("view-werk"),
  input: $("capture-input"),
  btnCapture: $("btn-capture"),
  list: $("thought-list"),
  count: $("thought-count"),
  empty: $("cocoon-empty"),
  btnClear: $("btn-clear"),
  btnDemo: $("btn-demo"),
  dock: $("morph-dock"),
  btnMorph: $("btn-morph"),
  morphSub: $("morph-sub"),
  morphing: $("morphing"),
  morphingText: $("morphing-text"),
  werkDoc: $("werk-doc"),
  btnBack: $("btn-back"),
  btnCopy: $("btn-copy"),
  btnLastWerk: $("btn-last-werk"),
  brandHome: $("brand-home"),
  btnSettings: $("btn-settings"),
  sheet: $("sheet-settings"),
  backdrop: $("sheet-backdrop"),
  apiKey: $("api-key"),
  btnKeyToggle: $("btn-key-toggle"),
  btnSheetSave: $("btn-sheet-save"),
  btnWipe: $("btn-wipe"),
  toast: $("toast"),
};

/* ——— Persistenz ——— */

function load(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function save(key, value) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* Speicher voll oder blockiert – die App läuft trotzdem weiter */
  }
}

/* ——— Gedanken ——— */

function addGedanke(text) {
  const trimmed = text.trim();
  if (!trimmed) return;
  gedanken.push({ id: crypto.randomUUID(), text: trimmed, zeit: Date.now() });
  save(STORE.gedanken, gedanken);
  renderGedanken();
}

function removeGedanke(id) {
  const node = el.list.querySelector(`[data-id="${id}"]`);
  const commit = () => {
    gedanken = gedanken.filter((g) => g.id !== id);
    save(STORE.gedanken, gedanken);
    renderGedanken();
  };
  if (node && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
    node.classList.add("leaving");
    node.addEventListener("animationend", commit, { once: true });
  } else {
    commit();
  }
}

function formatZeit(ts) {
  return new Intl.DateTimeFormat("de-DE", {
    weekday: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(ts));
}

function renderGedanken() {
  el.list.replaceChildren(
    ...gedanken.map((g) => {
      const li = document.createElement("li");
      li.className = "thought";
      li.dataset.id = g.id;

      const dot = document.createElement("span");
      dot.className = "thought-dot";

      const body = document.createElement("div");
      body.className = "thought-body";
      const p = document.createElement("p");
      p.className = "thought-text";
      p.textContent = g.text;
      const time = document.createElement("p");
      time.className = "thought-time";
      time.textContent = formatZeit(g.zeit);
      body.append(p, time);

      const remove = document.createElement("button");
      remove.className = "thought-remove";
      remove.setAttribute("aria-label", "Gedanken entfernen");
      remove.innerHTML =
        '<svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
      remove.addEventListener("click", () => removeGedanke(g.id));

      li.append(dot, body, remove);
      return li;
    })
  );

  const n = gedanken.length;
  el.count.textContent = n;
  el.empty.hidden = n > 0;
  el.btnClear.hidden = n === 0;
  el.dock.hidden = n === 0;
  el.morphSub.textContent = n === 1 ? "aus 1 Gedanken" : `aus ${n} Gedanken`;
  el.btnLastWerk.hidden = !werk;
}

/* ——— Ansichten ——— */

function showStudio() {
  el.viewWerk.hidden = true;
  el.viewStudio.hidden = false;
  window.scrollTo({ top: 0 });
}

function showWerk() {
  if (!werk) return;
  renderWerk(werk);
  el.viewStudio.hidden = true;
  el.viewWerk.hidden = false;
  window.scrollTo({ top: 0 });
}

/* ——— Meisterwerk rendern ——— */

function renderWerk(w) {
  const doc = el.werkDoc;
  doc.replaceChildren();

  const eyebrow = p("werk-eyebrow", "Meisterwerk");
  const title = document.createElement("h1");
  title.className = "werk-title";
  title.textContent = w.titel;
  const subtitle = p("werk-subtitle", w.untertitel);
  const essenz = p("werk-essenz", w.essenz);
  doc.append(eyebrow, title, subtitle, essenz);

  for (const thema of w.themen || []) {
    const sec = document.createElement("section");
    sec.className = "werk-thema";

    const head = document.createElement("div");
    head.className = "werk-thema-head";
    const symbol = document.createElement("span");
    symbol.className = "werk-thema-symbol";
    symbol.textContent = thema.symbol || "•";
    const name = document.createElement("h2");
    name.className = "werk-thema-name";
    name.textContent = thema.name;
    head.append(symbol, name);
    sec.append(head, p("werk-thema-kern", thema.kern));

    if (thema.gedanken?.length) {
      sec.append(p("werk-label", "Deine Gedanken"));
      const ul = document.createElement("ul");
      ul.className = "werk-quellen";
      for (const g of thema.gedanken) ul.append(li(g));
      sec.append(ul);
    }

    if (thema.schritte?.length) {
      sec.append(p("werk-label", "Entspannte Schritte"));
      const ol = document.createElement("ol");
      ol.className = "werk-schritte";
      for (const s of thema.schritte) ol.append(li(s));
      sec.append(ol);
    }

    doc.append(sec);
  }

  if (w.erste_schritte?.length) {
    doc.append(hr());
    const sec = document.createElement("section");
    sec.className = "werk-thema";
    const name = document.createElement("h2");
    name.className = "werk-thema-name";
    name.textContent = "Womit du anfängst";
    sec.append(name, p("werk-thema-kern", "Nicht alles auf einmal. Nur das hier."));
    const ol = document.createElement("ol");
    ol.className = "werk-schritte";
    for (const s of w.erste_schritte) ol.append(li(s));
    sec.append(ol);
    doc.append(sec);
  }

  if (w.impuls) {
    doc.append(hr(), p("werk-impuls", `„${w.impuls}“`));
  }

  const meta = p(
    "werk-meta",
    `Verwandelt am ${new Intl.DateTimeFormat("de-DE", { dateStyle: "long", timeStyle: "short" }).format(new Date(w.erstellt || Date.now()))}${w.demo ? " · Beispiel" : ""}`
  );
  doc.append(meta);

  function p(cls, text) {
    const node = document.createElement("p");
    node.className = cls;
    node.textContent = text;
    return node;
  }
  function li(text) {
    const node = document.createElement("li");
    node.textContent = text;
    return node;
  }
  function hr() {
    const node = document.createElement("hr");
    node.className = "werk-divider";
    return node;
  }
}

/* ——— Markdown-Export ——— */

function werkAlsMarkdown(w) {
  const lines = [`# ${w.titel}`, "", `*${w.untertitel}*`, "", `> ${w.essenz}`, ""];
  for (const t of w.themen || []) {
    lines.push(`## ${t.symbol} ${t.name}`, "", t.kern, "");
    if (t.gedanken?.length) {
      lines.push("**Deine Gedanken:**", "");
      for (const g of t.gedanken) lines.push(`- ${g}`);
      lines.push("");
    }
    if (t.schritte?.length) {
      lines.push("**Entspannte Schritte:**", "");
      t.schritte.forEach((s, i) => lines.push(`${i + 1}. ${s}`));
      lines.push("");
    }
  }
  if (w.erste_schritte?.length) {
    lines.push("## Womit du anfängst", "");
    w.erste_schritte.forEach((s, i) => lines.push(`${i + 1}. ${s}`));
    lines.push("");
  }
  if (w.impuls) lines.push(`— *${w.impuls}*`);
  return lines.join("\n");
}

/* ——— Metamorphose (Claude) ——— */

const MORPH_PHASEN = [
  "Die Metamorphose beginnt …",
  "Gedanken werden sortiert …",
  "Der rote Faden zeigt sich …",
  "Themen nehmen Form an …",
  "Gleich ist es so weit …",
];

async function metamorphose() {
  if (morphing || gedanken.length === 0) return;

  const key = load(STORE.schluessel, "");
  if (!key) {
    openSettings();
    toast("Für die Metamorphose brauchst du einen API-Schlüssel.");
    return;
  }

  morphing = true;
  el.morphing.hidden = false;
  let phase = 0;
  el.morphingText.textContent = MORPH_PHASEN[0];
  const phasenTimer = setInterval(() => {
    phase = Math.min(phase + 1, MORPH_PHASEN.length - 1);
    el.morphingText.textContent = MORPH_PHASEN[phase];
  }, 6000);

  try {
    const daten = await rufeClaude(gedanken, key);
    werk = { ...daten, erstellt: Date.now(), quelle: gedanken.map((g) => g.text) };
    save(STORE.werk, werk);
    gedanken = [];
    save(STORE.gedanken, gedanken);
    renderGedanken();
    showWerk();
    toast("Verwandelt. ✨");
  } catch (err) {
    toast(fehlerText(err));
  } finally {
    clearInterval(phasenTimer);
    el.morphing.hidden = true;
    morphing = false;
  }
}

async function rufeClaude(gedankenListe, apiKey) {
  const nutzerText = [
    "Hier sind die Gedanken aus meinem Kokon, in der Reihenfolge, in der ich sie hineingeworfen habe:",
    "",
    ...gedankenListe.map((g, i) => `${i + 1}. ${g.text}`),
    "",
    "Verwandle sie in ein Meisterwerk.",
  ].join("\n");

  const res = await fetch(API_URL, {
    method: "POST",
    headers: {
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
      "anthropic-beta": "server-side-fallback-2026-07-01",
      "anthropic-dangerous-direct-browser-access": "true",
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: 16000,
      stream: true,
      fallbacks: "default",
      system: SYSTEM_PROMPT,
      output_config: { format: { type: "json_schema", schema: WERK_SCHEMA } },
      messages: [{ role: "user", content: nutzerText }],
    }),
  });

  if (!res.ok) {
    let detail = "";
    try {
      detail = (await res.json())?.error?.message || "";
    } catch {
      /* Fehlertext ist optional */
    }
    const err = new Error(detail || `HTTP ${res.status}`);
    err.status = res.status;
    throw err;
  }

  const { text, stopReason } = await liesStream(res.body);

  if (stopReason === "refusal") {
    const err = new Error("refusal");
    err.refusal = true;
    throw err;
  }

  try {
    return JSON.parse(text);
  } catch {
    throw new Error("Die Antwort war kein gültiges JSON – bitte noch einmal versuchen.");
  }
}

async function liesStream(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let text = "";
  let stopReason = null;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop();

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice(6).trim();
      if (!payload || payload === "[DONE]") continue;

      let event;
      try {
        event = JSON.parse(payload);
      } catch {
        continue;
      }

      if (event.type === "content_block_delta" && event.delta?.type === "text_delta") {
        text += event.delta.text;
      } else if (event.type === "message_delta" && event.delta?.stop_reason) {
        stopReason = event.delta.stop_reason;
      } else if (event.type === "error") {
        throw new Error(event.error?.message || "Stream-Fehler");
      }
    }
  }

  return { text, stopReason };
}

function fehlerText(err) {
  if (err.refusal) return "Diese Anfrage wurde abgelehnt. Formuliere die Gedanken anders und versuch es erneut.";
  if (err.status === 401) return "Der API-Schlüssel wurde nicht akzeptiert. Prüf ihn in den Einstellungen.";
  if (err.status === 429) return "Gerade zu viele Anfragen. Atme einmal durch und versuch es gleich noch mal.";
  if (err.status >= 500) return "Die API ist kurz nicht erreichbar. Gleich noch einmal versuchen.";
  if (err instanceof TypeError) return "Keine Verbindung. Bist du online?";
  return err.message || "Etwas ist schiefgegangen. Deine Gedanken sind aber sicher.";
}

/* ——— Beispiel ——— */

const DEMO_GEDANKEN = [
  "workshop wo leute einfach entspannen lernen. yoga morgens?? oder joggen",
  "claude läuft im hintergrund und sammelt alles was einem beim kaffee einfällt",
  "es muss sich anfühlen wie shazam: handy raus, eine geste, fertig, weiterleben",
  "kein büro-vibe. eher wohnzimmer. pflanzen. gute tassen.",
  "am ende der woche bekommt jeder sein eigenes meisterwerk aus seinen losen ideen",
];

const DEMO_WERK = {
  demo: true,
  titel: "Das Studio, das für dich denkt",
  untertitel: "Ein Ort zum Entspannen, an dem Ideen nebenbei zu etwas Ganzem reifen.",
  essenz:
    "Alle Gedanken kreisen um dieselbe Umkehrung: Nicht der Mensch arbeitet für seine Ideen, die Ideen arbeiten für ihn. Der Ort fühlt sich an wie ein Wohnzimmer, der Ablauf wie Shazam – eine Geste, dann weiterleben. Die Technik bleibt unsichtbar, bis am Ende etwas Fertiges dasteht.",
  themen: [
    {
      name: "Der Ort",
      symbol: "🌿",
      kern: "Ein Studio, das nach Ankommen aussieht, nicht nach Arbeit – Bewegung am Morgen, Kaffee dazwischen, kein Büro-Vibe.",
      gedanken: [
        "Ein Workshop, in dem Leute einfach entspannen lernen – morgens Yoga oder Joggen.",
        "Kein Büro-Vibe: eher Wohnzimmer, Pflanzen, gute Tassen.",
      ],
      schritte: [
        "Einen Probe-Vormittag mit fünf Freunden machen: Yoga, Kaffee, nichts weiter.",
        "Drei Räume anschauen, die sich wie Wohnzimmer anfühlen.",
        "Eine Tasse kaufen, die den Ton für alles Weitere setzt.",
      ],
    },
    {
      name: "Die unsichtbare Hilfe",
      symbol: "🫧",
      kern: "Claude sammelt im Hintergrund, was beim Kaffee so fällt – ohne dass jemand ein Gerät bedienen muss.",
      gedanken: [
        "Claude läuft im Hintergrund und sammelt alles, was einem beim Kaffee einfällt.",
        "Es muss sich anfühlen wie Shazam: Handy raus, eine Geste, fertig, weiterleben.",
      ],
      schritte: [
        "Die Ein-Gesten-Aufnahme mit dieser App eine Woche selbst testen.",
        "Notieren, an welchen Stellen es sich noch nach Bedienung anfühlt.",
      ],
    },
    {
      name: "Das Versprechen",
      symbol: "🎁",
      kern: "Am Ende steht etwas Fertiges – jeder geht mit seinem eigenen Meisterwerk aus losen Ideen nach Hause.",
      gedanken: ["Am Ende der Woche bekommt jeder sein eigenes Meisterwerk aus seinen losen Ideen."],
      schritte: [
        "Ein Beispiel-Meisterwerk gestalten, das man in die Hand nehmen kann.",
        "Einen Satz finden, der das Versprechen erklärt, ohne Technik zu erwähnen.",
      ],
    },
  ],
  erste_schritte: [
    "Den Probe-Vormittag terminieren – ein Datum genügt, der Rest ergibt sich.",
    "Diese App eine Woche als eigenes Ideen-Zuhause benutzen.",
    "Das erste Beispiel-Meisterwerk ausdrucken und ins künftige Studio legen.",
  ],
  impuls: "Die besten Ideen kommen nicht am Schreibtisch. Sie kommen, wenn niemand mehr mitschreiben muss.",
};

function zeigeDemo() {
  const jetzt = Date.now();
  gedanken = DEMO_GEDANKEN.map((text, i) => ({
    id: crypto.randomUUID(),
    text,
    zeit: jetzt - (DEMO_GEDANKEN.length - i) * 47 * 60 * 1000,
  }));
  save(STORE.gedanken, gedanken);
  renderGedanken();
  werk = { ...DEMO_WERK, erstellt: jetzt, quelle: DEMO_GEDANKEN };
  save(STORE.werk, werk);
  showWerk();
  toast("Ein Beispiel – deine eigenen Gedanken warten im Kokon.");
}

/* ——— Einstellungen ——— */

function openSettings() {
  el.apiKey.value = load(STORE.schluessel, "");
  el.apiKey.type = "password";
  el.btnKeyToggle.textContent = "Zeigen";
  el.sheet.hidden = false;
  el.backdrop.hidden = false;
  el.apiKey.focus();
}

function closeSettings() {
  const key = el.apiKey.value.trim();
  save(STORE.schluessel, key || null);
  el.sheet.hidden = true;
  el.backdrop.hidden = true;
}

/* ——— Toast ——— */

let toastTimer;
function toast(text) {
  el.toast.textContent = text;
  el.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.toast.hidden = true;
  }, 3600);
}

/* ——— Eingabe ——— */

function autoGrow() {
  el.input.style.height = "auto";
  el.input.style.height = `${el.input.scrollHeight}px`;
}

function capture() {
  addGedanke(el.input.value);
  el.input.value = "";
  autoGrow();
  el.input.focus();
}

/* ——— Ereignisse ——— */

el.input.addEventListener("input", autoGrow);
el.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    capture();
  }
});
el.btnCapture.addEventListener("click", capture);

el.btnClear.addEventListener("click", () => {
  if (!confirm("Alle Gedanken aus dem Kokon entfernen?")) return;
  gedanken = [];
  save(STORE.gedanken, gedanken);
  renderGedanken();
});

el.btnDemo.addEventListener("click", zeigeDemo);
el.btnMorph.addEventListener("click", metamorphose);

el.btnBack.addEventListener("click", showStudio);
el.brandHome.addEventListener("click", (e) => {
  e.preventDefault();
  showStudio();
});
el.btnLastWerk.addEventListener("click", showWerk);

el.btnCopy.addEventListener("click", async () => {
  if (!werk) return;
  try {
    await navigator.clipboard.writeText(werkAlsMarkdown(werk));
    toast("Als Markdown kopiert.");
  } catch {
    toast("Kopieren hat nicht geklappt.");
  }
});

el.btnSettings.addEventListener("click", openSettings);
el.btnSheetSave.addEventListener("click", closeSettings);
el.backdrop.addEventListener("click", closeSettings);

el.btnKeyToggle.addEventListener("click", () => {
  const versteckt = el.apiKey.type === "password";
  el.apiKey.type = versteckt ? "text" : "password";
  el.btnKeyToggle.textContent = versteckt ? "Verbergen" : "Zeigen";
});

el.btnWipe.addEventListener("click", () => {
  if (!confirm("Wirklich alles löschen? Gedanken, Meisterwerk und Schlüssel.")) return;
  Object.values(STORE).forEach((k) => localStorage.removeItem(k));
  gedanken = [];
  werk = null;
  closeSettings();
  renderGedanken();
  showStudio();
  toast("Alles gelöscht. Frischer Kokon.");
});

document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    showStudio();
    el.input.focus();
  }
  if (e.key === "Escape" && !el.sheet.hidden) closeSettings();
});

/* ——— Start ——— */

renderGedanken();
autoGrow();
