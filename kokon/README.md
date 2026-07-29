# Kokon 🌿

**Leb dein Leben. Der Kokon arbeitet.**

Kokon ist ein ruhiger Ort für lose Gedanken. Zwischen Yoga, Laufen und Kaffee wirfst du mit einer Geste etwas hinein – einen halben Satz, eine vage Idee – und lebst weiter. Wenn du bereit bist, verwandelt die **Metamorphose** (Claude Opus 5) alle Gedanken in ein **Meisterwerk**: Essenz, Themen und entspannte erste Schritte.

Das Prinzip ist von Shazam geliehen: eine Geste, kein Nachdenken über das Werkzeug, sofort zurück ins Leben.

## Starten

Kein Build, keine Abhängigkeiten – drei statische Dateien.

```bash
cd kokon
python3 -m http.server 8000
# → http://localhost:8000
```

Oder `index.html` direkt im Browser öffnen.

## Metamorphose einrichten

1. Zahnrad oben rechts öffnen.
2. Anthropic API-Schlüssel eintragen ([console.anthropic.com](https://console.anthropic.com)).
3. Gedanken hineinwerfen, **Metamorphose** drücken.

Ohne Schlüssel lässt sich über „Beispiel ansehen" ein fertiges Meisterwerk erleben.

## Wie es funktioniert

- **Aufnehmen:** `⏎` legt ab, `⇧⏎` macht eine neue Zeile, `⌘K` fokussiert das Feld von überall.
- **Kokon:** Alle Gedanken warten lokal (localStorage) – nichts verlässt das Gerät, bis du die Metamorphose startest.
- **Metamorphose:** Ein Aufruf an die Anthropic-API (Claude Opus 5, Streaming, strukturierte Ausgabe per JSON-Schema, Server-Fallback bei Ablehnungen). Der Schlüssel geht direkt vom Browser an die API – kein Server dazwischen.
- **Meisterwerk:** Wird gerendert, gespeichert und lässt sich als Markdown kopieren.

## Datenschutz

Gedanken, Meisterwerke und der API-Schlüssel liegen ausschließlich im localStorage deines Browsers. „Alles löschen" in den Einstellungen entfernt sie rückstandslos.

## Design

Bewusst nah an Apples Ruhe gebaut: Systemschrift, viel Weißraum, eine Akzentfarbe (Salbeigrün), weiche Tiefe, Licht- und Dunkelmodus folgen dem System, Bewegung respektiert `prefers-reduced-motion`.
