"use client";

import MagicBento from "./MagicBento";

export default function TelemetryBento({
  metrics,
}: {
  metrics: {
    id: string;
    label: string;
    title: string;
    description: string;
  }[];
}) {
  return (
    <MagicBento
      items={metrics}
      columns={4}
      rowHeight={154}
      gap={12}
      radius={20}
      enableBorderGlow
      enableSpotlight
      spotlightRadius={300}
      clickEffect
      enableStars={false}
      particleCount={24}
      enableTilt={false}
      enableMagnetism={false}
      textAutoHide
      intro
      className="telemetry-bento"
    />
  );
}
