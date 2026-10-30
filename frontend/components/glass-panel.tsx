"use client";

import type { CSSProperties, ReactNode } from "react";
import GlassSurface from "./GlassSurface";

export default function GlassPanel({
  children,
  className = "",
  style,
}: {
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
}) {
  return (
    <GlassSurface
      width="100%"
      height="100%"
      borderRadius={14}
      borderWidth={0.05}
      backgroundOpacity={0.1}
      distortionScale={-90}
      displace={0.25}
      blur={8}
      renderer="svg"
      className={`glass-panel ${className}`}
      style={style}
    >
      {children}
    </GlassSurface>
  );
}
