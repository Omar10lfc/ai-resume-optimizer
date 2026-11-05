import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * Shared markdown renderer, token-styled once so every tab in the results
 * dashboard renders identically without duplicating the component map.
 */

const components = {
  h1: (props: React.ComponentProps<"h1">) => (
    <h1 className="text-base font-bold text-ink mt-4 mb-3 pb-1 border-b border-line" {...props} />
  ),
  h2: (props: React.ComponentProps<"h2">) => (
    <h2 className="text-sm font-semibold text-ink mt-5 mb-2 pb-1 border-b border-line" {...props} />
  ),
  h3: (props: React.ComponentProps<"h3">) => (
    <h3 className="text-xs font-semibold uppercase tracking-wide text-accent mt-4 mb-2" {...props} />
  ),
  p: (props: React.ComponentProps<"p">) => (
    <p className="text-xs text-ink-2 leading-relaxed my-2" {...props} />
  ),
  strong: (props: React.ComponentProps<"strong">) => (
    <strong className="font-semibold text-ink" {...props} />
  ),
  blockquote: (props: React.ComponentProps<"blockquote">) => (
    <blockquote
      className="border-l-2 border-accent bg-raised/60 pl-3 py-2 my-2 rounded-r-md2 text-2xs text-ink-2 italic"
      {...props}
    />
  ),
  ul: (props: React.ComponentProps<"ul">) => (
    <ul className="list-disc pl-5 my-2 space-y-1.5 text-xs text-ink-2" {...props} />
  ),
  ol: (props: React.ComponentProps<"ol">) => (
    <ol className="list-decimal pl-5 my-2 space-y-1.5 text-xs text-ink-2" {...props} />
  ),
  li: (props: React.ComponentProps<"li">) => (
    <li className="text-xs text-ink-2 leading-relaxed" {...props} />
  ),
  code: (props: React.ComponentProps<"code">) => (
    <code className="font-mono text-2xs px-1.5 py-0.5 rounded-sm2 bg-raised text-accent border border-line" {...props} />
  ),
  table: (props: React.ComponentProps<"table">) => (
    <div className="overflow-x-auto my-3">
      <table className="w-full text-2xs border-collapse" {...props} />
    </div>
  ),
  th: (props: React.ComponentProps<"th">) => (
    <th className="text-left font-semibold text-ink border-b border-line-strong px-2 py-1.5" {...props} />
  ),
  td: (props: React.ComponentProps<"td">) => (
    <td className="text-ink-2 border-b border-line px-2 py-1.5" {...props} />
  ),
  a: (props: React.ComponentProps<"a">) => (
    <a className="text-accent underline underline-offset-2 hover:text-accent-strong" target="_blank" rel="noreferrer" {...props} />
  ),
};

export const Markdown: React.FC<{ children: string; className?: string }> = ({ children, className = "" }) => (
  <div className={`max-w-none leading-relaxed ${className}`}>
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
      {children}
    </ReactMarkdown>
  </div>
);
