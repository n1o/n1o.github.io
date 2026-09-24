import { glob } from "astro/loaders";
import { defineCollection, z } from "astro:content";

const posts = defineCollection({
  loader: glob({ pattern: "*.md", base: "./src/content/posts" }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    description: z.string().optional(),
    draft: z.boolean().default(false),
    tags: z.array(z.string()).default([]),
    categories: z.array(z.string()).default([]),
    series: z.array(z.string()).default([]),
  }),
});

const pages = defineCollection({
  loader: glob({ pattern: "*.md", base: "./src/content" }),
  schema: z.object({
    title: z.string().optional(),
    date: z.coerce.date().optional(),
    description: z.string().optional(),
  }),
});

export const collections = { posts, pages };
