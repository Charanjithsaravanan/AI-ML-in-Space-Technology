import type { CSSProperties, ReactNode } from "react";

type MagicBentoItem = {
  id?: string;
  title?: string;
  description?: string;
  label?: string;
  icon?: ReactNode;
  visual?: ReactNode | string;
  content?: ReactNode;
  colSpan?: number;
  rowSpan?: number;
  href?: string;
  onClick?: (event: unknown) => void;
};

type MagicBentoProps = {
  items?: MagicBentoItem[];
  columns?: number;
  rowHeight?: number;
  gap?: number;
  radius?: number;
  theme?: "dark" | "light";
  glowColor?: string;
  textAutoHide?: boolean;
  enableBorderGlow?: boolean;
  enableSpotlight?: boolean;
  spotlightRadius?: number;
  enableStars?: boolean;
  particleCount?: number;
  enableTilt?: boolean;
  enableMagnetism?: boolean;
  clickEffect?: boolean;
  intro?: boolean;
  disableAnimations?: boolean;
  className?: string;
  style?: CSSProperties;
};

declare const MagicBento: (props: MagicBentoProps) => JSX.Element;

export default MagicBento;
