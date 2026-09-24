import { defineConfig } from "astro/config";

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
