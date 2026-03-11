import type { Metadata } from "next";
import "./globals.css";
import { AppProvider } from "@/components/providers";

export const metadata: Metadata = {
  title: "Omniverse — Analysis Validation",
  description: "Clinical research analysis validation platform",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">
        <AppProvider>{children}</AppProvider>
      </body>
    </html>
  );
}
