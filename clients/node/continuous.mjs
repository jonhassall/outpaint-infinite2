#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const api = (process.env.API_URL || "http://localhost:8000").replace(/\/$/, "");
const imagePath = process.env.IMAGE || process.argv[2];
const prompt = process.env.PROMPT || process.argv.slice(3).join(" ");
const apiKey = process.env.API_KEY || "";

if (!imagePath || !prompt) {
  console.error('Usage: IMAGE=input.png PROMPT="full scene prompt" node continuous.mjs');
  process.exit(2);
}

const headers = apiKey ? { "x-api-key": apiKey } : {};
const form = new FormData();
const bytes = fs.readFileSync(imagePath);
form.append("image", new Blob([bytes]), path.basename(imagePath));
form.append("prompt", prompt);
form.append("directions", process.env.DIRECTIONS || "left,right,up,down");
form.append("expand_pixels", process.env.EXPAND_PIXELS || "256");
form.append("steps", process.env.STEPS || "8");
form.append("max_steps", process.env.MAX_STEPS || "20");
form.append("delay_seconds", process.env.DELAY_SECONDS || "0");
form.append("randomize_seed", process.env.RANDOMIZE_SEED || "true");
form.append("seed", process.env.SEED || "42");

const start = await fetch(`${api}/api/v1/jobs/continuous`, {
  method: "POST",
  headers,
  body: form,
});
if (!start.ok) throw new Error(await start.text());
const job = await start.json();
console.log("Started job", job.id);

const events = await fetch(`${api}${job.events_url}`, { headers });
if (!events.ok || !events.body) throw new Error(await events.text());

const decoder = new TextDecoder();
let buffer = "";
for await (const chunk of events.body) {
  buffer += decoder.decode(chunk, { stream: true });
  const frames = buffer.split("\n\n");
  buffer = frames.pop() || "";
  for (const frame of frames) {
    const dataLine = frame.split("\n").find((line) => line.startsWith("data: "));
    if (!dataLine) continue;
    const update = JSON.parse(dataLine.slice(6));
    console.log(
      `[${update.status}] step=${update.current_step} direction=${update.latest_direction || "-"} image=${update.latest_url || "-"}`,
    );
  }
}
