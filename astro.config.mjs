import { defineConfig } from "astro/config";

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
