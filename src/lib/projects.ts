// Single source of truth for project cards.
// Flags mirror `features` in astro.config.mjs — flip them there to
// include/exclude a card from the homepage and /projects/ at build time.
type ProjectFlag = "hathor" | "dissectingAi" | "studyNotes";
type Project = {
  flag: ProjectFlag;
  href: string;
  category: string;
  name: string;
  claim: string;
  pitch: string;
  tag: string;
  cta: string;
};

const features: Record<ProjectFlag, boolean> = {
  hathor: true, // hathortts.com — TTS product
  dissectingAi: false, // dissecting-ai.dev — AI learning platform
  studyNotes: true, // n1o.github.io/study_notes
};

const all: Project[] = [
  {
    flag: "hathor",
    href: "https://hathortts.com",
    category: "AUDIO",
    name: "Hathor",
    claim: "Turn entire books into audio for the price of a coffee.",
    pitch:
      "Paste text, upload an EPUB, or drop a JSONL batch — Hathor turns it into natural, high-fidelity speech. Voice cloning, voice design, and an HTTP API, at a fraction of the usual cost.",
    tag: "TEXT → AUDIO",
    cta: "Visit hathortts.com",
  },
  {
    flag: "dissectingAi",
    href: "https://dissecting-ai.dev",
    category: "AI RESEARCH",
    name: "Dissecting AI",
    claim: "Papers → summary · lecture · quiz · audio.",
    pitch:
      "A learning platform for cutting-edge AI research. Curated paths, topic deep-dives, and a public library — submit any paper and get it turned into a lesson. First module free, no account needed to browse.",
    tag: "PAPERS → UNDERSTANDING",
    cta: "Visit dissecting-ai.dev",
  },
  {
    flag: "studyNotes",
    href: "https://n1o.github.io/study_notes/content.html",
    category: "REFERENCE",
    name: "Study notes",
    claim: "Thousands of pages of notes, rewritten to stay useful.",
    pitch:
      "My compilation of notes from various subjects — continuously migrated, rewritten, and simplified. Zettelkasten-style: short, easy to follow, one subject per note.",
    tag: "NOTES → KNOWLEDGE",
    cta: "Browse the notes",
  },
];

export const projects = all.filter((p) => features[p.flag]);
