"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * Normalize common LLM output patterns into valid markdown so lists and
 * newlines render reliably.
 */
function normalizeMarkdown(md: string): string {
  // Normalize line endings and Unicode bullet markers into markdown list items.
  return md
    .replace(/\r\n?/g, "\n")
    .replace(/^[ \t]*[•◦▪‣∙][ \t]+/gm, "- ")
    .replace(/^([ \t]*\d+)\)[ \t]+/gm, "$1. ")
    // Some model outputs emit list markers on their own line before content:
    // "-\nItem text" or "1.\n\n  Item text" -> "- Item text" / "1. Item text"
    .replace(/^([ \t]*\d+\.)[ \t]*\n(?:[ \t]*\n)*[ \t]*/gm, "$1 ")
    .replace(/^([ \t]*[-*+])[ \t]*\n(?:[ \t]*\n)*[ \t]*/gm, "$1 ");
}

export function Markdown({ content }: { content: string }) {
  return (
    <div className="prose-chat">
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ children }) => (
          <p className="mb-2 last:mb-0 leading-relaxed whitespace-pre-wrap">{children}</p>
        ),
        strong: ({ children }) => (
          <strong className="font-semibold">{children}</strong>
        ),
        em: ({ children }) => <em className="italic">{children}</em>,
        ul: ({ children }) => <ul>{children}</ul>,
        ol: ({ children }) => <ol>{children}</ol>,
        li: ({ children }) => <li className="leading-relaxed whitespace-pre-wrap">{children}</li>,
        code: ({ children, className }) => {
          const isBlock = className?.includes("language-");
          if (isBlock) {
            return (
              <pre className="bg-background/60 border border-border rounded-lg px-3 py-2.5 my-2 overflow-x-auto text-xs font-mono">
                <code>{children}</code>
              </pre>
            );
          }
          return (
            <code className="bg-background/60 border border-border rounded px-1.5 py-0.5 text-xs font-mono">
              {children}
            </code>
          );
        },
        h1: ({ children }) => (
          <h1 className="text-base font-bold mb-2 mt-4 first:mt-0 pb-1 border-b border-border/60">
            {children}
          </h1>
        ),
        h2: ({ children }) => (
          <h2 className="text-sm font-semibold mb-2 mt-4 first:mt-0 pb-1 border-b border-border/60">
            {children}
          </h2>
        ),
        h3: ({ children }) => (
          <h3 className="text-sm font-medium mb-1.5 mt-3 first:mt-0">
            {children}
          </h3>
        ),
        blockquote: ({ children }) => (
          <blockquote className="border-l-2 border-muted-foreground/30 pl-3 my-2 text-muted-foreground italic">
            {children}
          </blockquote>
        ),
        hr: () => <hr className="my-4 border-border/60" />,
        // Wrap table in a scrollable container so wide tables don't overflow the chat bubble
        table: ({ children }) => (
          <div className="overflow-x-auto my-3 rounded-lg border border-border">
            <table className="w-full text-xs border-collapse min-w-[400px]">
              {children}
            </table>
          </div>
        ),
        thead: ({ children }) => (
          <thead className="bg-background/60">{children}</thead>
        ),
        tbody: ({ children }) => (
          <tbody className="divide-y divide-border">{children}</tbody>
        ),
        tr: ({ children }) => (
          <tr className="even:bg-background/30 transition-colors">{children}</tr>
        ),
        th: ({ children }) => (
          <th className="px-3 py-2 text-left font-semibold text-foreground/80 border-b border-border whitespace-nowrap">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="px-3 py-2 text-foreground/90 align-top">
            {children}
          </td>
        ),
      }}
    >
      {normalizeMarkdown(content)}
    </ReactMarkdown>
    </div>
  );
}
