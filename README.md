# LLM Inference

A static, dependency-free website that explains how serving engines run large
language models across GPUs: continuous batching and the paged KV cache on one
GPU, then tensor, pipeline, data and expert parallelism across many, with
interactive visualizations and general guidance on which split suits which use case.
Examples and defaults come from vLLM.

## Run it

```bash
cd site
python3 -m http.server 8787 --bind 0.0.0.0
# open http://<host>:8787/
```

Any static file server works; there is no build step.

## Layout

```
site/
  index.html          content, citations (data-cite="key" -> numbered sources)
  css/style.css       design tokens (light + dark), layout
  js/common.js        theme, canvas helpers, citations
  js/viz-bus.js       the bus loop: animated analogy for decode, batching and the KV cache
  js/glossary.js      glossary search and filters, inline term links with hover definitions
  js/viz-batching.js  continuous batching simulation with paged KV grid
  js/viz-split.js     split visualizer: how TP / PP / DP / EP lay out a model
```

## Editing content

- To cite a source, add `<span data-cite="key"></span>` in the text and a
  `<li id="src-key">` entry in the Sources list. Numbers are assigned at load.
