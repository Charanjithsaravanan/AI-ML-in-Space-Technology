import { copyFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";

const root = resolve(import.meta.dirname, "../..");
const files = [
  ["results/benchmark.json", "public/data/benchmark.json"],
  ["results/robustness.json", "public/data/robustness.json"],
  ["runs/cloud_screener.json", "public/data/cloud_screener.json"],
];

for (const [source, destination] of files) {
  const target = resolve(import.meta.dirname, "..", destination);
  await mkdir(dirname(target), { recursive: true });
  await copyFile(resolve(root, source), target);
  console.log(`synced ${source} -> ${destination}`);
}
