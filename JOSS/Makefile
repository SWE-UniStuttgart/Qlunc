# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
FILENAME = JOSS_draft

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

pdf:
    mkdir $(BUILDDIR) -p  # Creates the BUILDDIR if it doesn't already exist.
    pandoc $(FILENAME).md \
    --filter pandoc-citeproc \
    --from=markdown+tex_math_single_backslash+tex_math_dollars+raw_tex \
    --to=latex \
    --output=$(BUILDDIR)/$(FILENAME).pdf \
    --pdf-engine=xelatex
	
html:
    mkdir $(BUILDDIR) -p  # Creates the BUILDDIR if it doesn't already exist.
    pandoc $(FILENAME).md \
    --filter pandoc-citeproc \
    --from=markdown+tex_math_single_backslash+tex_math_dollars \
    --to=html5 \
    --output=$(BUILDDIR)/$(FILENAME).html \
    --mathjax \
    --self-contained