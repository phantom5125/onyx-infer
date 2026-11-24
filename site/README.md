# Project site and engineering blog

The `site/` directory serves a static site with no framework or build-time dependencies. GitHub Pages uploads the checked-in tree.

## Files

- `index.html` contains the landing page.
- `assets/` contains shared CSS, the site mark, and compressed generation previews in `gallery/`. The landing page and both README files reference these curated images; original PNGs remain generation evidence rather than web assets.
- `blog/posts/` contains the Markdown sources. `tools/build_blog.py` generates the blog index and individual HTML pages.

The repository root `assets/` directory contains the README banner.

## Publish a post

Add a Markdown file under `site/blog/posts/`. Set `title`, `date`, and `summary` in its frontmatter. Optional fields include `image` relative to `site/`, `caption`, `tags`, and `lang` for posts written in a language other than English. Generate the HTML and preview the site locally.

```bash
uv run python tools/build_blog.py
python3 -m http.server 8000 -d site
```

Commit the Markdown and generated HTML together. The generator also supports `--watch`.

## Deploy

Select **GitHub Actions** as the source in the repository's Pages settings, then dispatch the Pages workflow. Set `ONYX_DEPLOY_PAGES` to `true` to enable automatic deploys after changes to site files on `main`.
