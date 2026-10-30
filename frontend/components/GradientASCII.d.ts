import type { CSSProperties, HTMLAttributes } from "react";

declare type GradientASCIIProps = Omit<HTMLAttributes<HTMLDivElement>, "color"> & {
  colors?: string[];
  shape?: "horizon" | "pinch" | "rise" | "eclipse";
  characters?: string;
  cellSize?: number;
  density?: number;
  glyphOpacity?: number;
  glow?: number;
  depthOfField?: number;
  flicker?: number;
  softness?: number;
  scale?: number;
  flow?: number;
  speed?: number;
  gradientOpacity?: number;
  grain?: number;
  variation?: number;
  mouseInteraction?: boolean;
  scrambleRadius?: number;
  clickEffect?: "none" | "wave" | "burst";
  intro?: boolean;
  lightMode?: boolean;
  fontFamily?: string;
  dpr?: number;
  className?: string;
  style?: CSSProperties;
};

declare const GradientASCII: (props: GradientASCIIProps) => JSX.Element;

export default GradientASCII;
