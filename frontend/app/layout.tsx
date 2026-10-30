import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GEOSAT Mission Dashboard",
  description: "Read-only retrieval dashboard for GEOSAT model outputs.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
