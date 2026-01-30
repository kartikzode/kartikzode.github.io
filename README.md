# Thombya's Technical Blog

A personal blog built with [Hugo](https://gohugo.io/) and the [PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme, hosted on GitHub Pages.

## About

This blog focuses on technical topics including:
- CUDA Programming
- GEMM Optimizations
- Reinforcement Learning

## 🚀 Quick Start

### Prerequisites
- [Hugo Extended](https://gohugo.io/installation/) v0.146.0 or higher
- Git

### Local Development

1. Clone the repository with submodules:
```bash
git clone --recurse-submodules https://github.com/kartikzode/thombya.github.io.git
cd thombya.github.io
```

2. If you already cloned without submodules, initialize them:
```bash
git submodule update --init --recursive
```

3. Start the development server:
```bash
hugo server -D
```

4. Open your browser to `http://localhost:1313/thombya.github.io/`

## 📝 Creating New Posts

Create a new blog post:
```bash
hugo new posts/my-new-post.md
```

Edit the frontmatter and content in `content/posts/my-new-post.md`:
```yaml
---
title: "My New Post"
date: 2024-01-30T10:00:00-00:00
draft: false
tags: ["tag1", "tag2"]
categories: ["Category"]
author: "Thombya"
showToc: true
TocOpen: false
description: "A brief description"
---

Your content here...
```

## 🏗️ Building the Site

Build the static site:
```bash
hugo --minify
```

The generated site will be in the `public/` directory.

## 🚢 Deployment

The site automatically deploys to GitHub Pages when you push to the `main` branch via GitHub Actions.

### Setting up GitHub Pages

1. Go to your repository settings
2. Navigate to "Pages" in the sidebar
3. Under "Build and deployment", select "GitHub Actions" as the source
4. The workflow will automatically deploy your site

## 📁 Project Structure

```
.
├── archetypes/          # Content templates
├── config.yml           # Hugo configuration
├── content/            
│   ├── about.md        # About page
│   ├── posts/          # Blog posts
│   └── search.md       # Search page
├── themes/
│   └── PaperMod/       # Theme (git submodule)
├── .github/
│   └── workflows/
│       └── hugo.yml    # GitHub Actions deployment
└── public/             # Generated site (gitignored)
```

## 🎨 Customization

### Site Configuration

Edit `config.yml` to customize:
- Site title and description
- Author information
- Social media links
- Menu items
- Theme settings

### Theme Settings

The PaperMod theme offers many customization options. See the [PaperMod documentation](https://github.com/adityatelange/hugo-PaperMod/wiki/Features) for details.

## 📄 License

Content: © 2026 Thombya
Theme: MIT License (PaperMod)

## 🔗 Links

- [Live Site](https://kartikzode.github.io/thombya.github.io/)
- [Hugo Documentation](https://gohugo.io/documentation/)
- [PaperMod Theme](https://github.com/adityatelange/hugo-PaperMod)
# Update to trigger rebuild
