import { readdirSync } from "node:fs";
import { defineConfig } from "astro/config";

// Posts now live on dissecting-ai.dev (same slugs under /writing). GitHub Pages
// cannot send server-side 301s, so Astro emits a stub page per URL with
// <link rel="canonical">, an instant meta refresh and a JS fallback.
const NEW_SITE = "https://dissecting-ai.dev";
const postRedirects = Object.fromEntries(
  readdirSync("./src/content/posts")
    .filter((f) => f.endsWith(".md") || f.endsWith(".mdx"))
    .map((f) => f.replace(/\.mdx?$/, ""))
    .map((slug) => [`/posts/${slug}`, `${NEW_SITE}/writing/${slug}`]),
);

// Feature flags — flip to toggle what gets rendered into the site.
// Consumers: src/lib/projects.ts (projects shown on homepage + /projects/).
const features = {
  hathor: true, // hathortts.com — TTS product
  dissectingAi: false, // dissecting-ai.dev — AI learning platform
  studyNotes: true, // n1o.github.io/study_notes
};

// https://astro.build/config
export default defineConfig({
  site: "https://n1o.github.io",
  redirects: { "/posts": `${NEW_SITE}/writing`, ...postRedirects },
  markdown: {
    remarkPlugins: [["remark-math", { singleDollarTextMath: true }]],
    rehypePlugins: [
      ["rehype-katex", { output: "html", strict: false, throwOnError: false }],
    ],
    shikiConfig: {
      theme: "github-dark-default",
    },
  },
});
