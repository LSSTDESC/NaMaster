# To make the document with the default settings:
# > make
#
# To make with a specific format:
# > make <format>
#
# To tar up a flat version of a specific format:
# > make <format> tar
#
# Alex Drlica-Wagner: https://github.com/LSSTDESC/start_paper/issues/new?body=@kadrlica

# Primary file names - avoid cookiecutter variables, to enable `make
# upgrade` to cleanly over-write this Makefile...
main ?= main
#default=$(shell cat .metadata.json | grep 'default_format' | cut -d'"' -f4)
outname ?= $(notdir $(shell pwd))

#ifeq ($(default), {{ cookiecutter.default_format }})
style ?= lsstdescnote
#else
# style=${default}
#endif

# 'note' shall be an alias for 'lsstdescnote'
ifeq ($(style),note)
style = lsstdescnote
endif
# mkauthlist doesn't recognize lsstdescnote as a style, so tell it to do something generic
ifeq ($(style),lsstdescnote)
mastyle = tex
else
mastyle = $(style)
endif

DESCTEX := desc-tex
DESCTEXORIGIN := https://github.com/LSSTDESC/desc-tex.git
LSSTTEX := lsst-texmf
LSSTTEXORIGIN := https://github.com/lsst/lsst-texmf.git

localpip ?= F
PIPOPTS = --upgrade-strategy only-if-needed
ifeq ($(localpip),T)
MKAUTHBIN := bin/mkauthlist
MKAUTH := PYTHONPATH=$$(ls -d lib/python*/site-packages 2>/dev/null | head -n1):$$PYTHONPATH $(MKAUTHBIN)
PIPOPTS += --install-option="--prefix=$(PWD)"
GETMKAUTH := [ -e $(MKAUTHBIN) ] || pip install $(PIPOPTS) mkauthlist
else
MKAUTHBIN :=
MKAUTH := mkauthlist 
GETMKAUTH := pip list 2>/dev/null | grep -q mkauthlist || pip install $(PIPOPTS) mkauthlist
endif

# LATEX environment variables
export TEXINPUTS:=./$(DESCTEX)/styles/:./tables/:
export BSTINPUTS:=./$(DESCTEX)/bst/:
export BIBINPUTS:=./$(DESCTEX)/bib/:./$(LSSTTEX)/texmf/bibtex/bib/:

# LaTeX journal class switcher flags
# apj=\def\flag{apj}
# apjl=\def
# mnras=\def\flag{mnras}

# Submission flags (these need some thought)
# arxiv=\def\flag{emulateapj}
# submit=${aastex}
# draft=\def\linenums{\linenumbers}

# Files to copy when making tarball
tardir := tmp
figdir ?= ./figures
figures ?= $(figdir)/*.{png,jpg,pdf}
tabdir ?= ./tables
tables ?= $(tabdir)/*.tex
styles ?= ./$(DESCTEX)/styles/*.{sty,cls}
bibs ?= ./$(DESCTEX)/bib/*.bst
source = $(main).{tex,bbl,bib} $(DESCTEX)/bib/lsstdesc.bib $(DESCTEX)/ack/*.tex authors.tex contributions.tex $(LSSTTEX)/texmf/bibtex/bib/lsst.bib $(LSSTTEX)/texmf/bibtex/bib/lsst-dm.bib

tarfiles = $(figures) $(tables) $(styles) $(bibs) $(source)

# prefix for difference files
DIFPRE ?= diff_
# git repo branch to compare changes against
BASEBRANCH ?= master
# git repo branch we are compiling
THISBRANCH := $(shell git branch --no-color 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ \1/' | tr -d ' ')
# temp file for the version to compare against
DRAFT = $(main).$(BASEBRANCH)
# base for difference files to generate
DIFF = $(DIFPRE)$(THISBRANCH)--$(BASEBRANCH)


maketargets := all authlist clean copy diff help $(main) update tar templates tidy touch
.PHONY: $(maketargets)

styleopts := aastex61 apj apjl emulateapj lsstdescnote mnras note prd prl tex 
help:
	@echo "Usage: make [style=lsstdescnote] [localpip=F] <target(s)>\n Possible targets: $(maketargets) <style> $(DESCTEX) $(LSSTTEX)\n  all: equivalent to 'make $(main) copy'\n  authlist: use mkauthlist to generate latex author/affiliation listing\n  clean: remove latex temporary files AND compiled outputs\n  copy: copy $(main).pdf to $(outname).pdf\n  diff: if we're in a git repo, use latexdiff to compare the current $(main).tex vs the $(BASEBRANCH) branch, and attempt to compile to $(DIFF).pdf\n  help: what you're looking at\n  $(main): compile $(main).tex\n  update: update desc-tex, mkauthlist, and templates repo\n  tar: tar up latex source files to $(outname).tar.gz\n  templates: download latest templates to templates/\n  tidy: delete latex temporary files\n  touch: touch $(main).tex to force a recompile\n  <style>: equivalent to 'make style=<style> $(main) copy tar' (see style option)\n  $(DESCTEX): download $(DESCTEX), which provides the DESC header logo and latex helper files (done automatically on first latex compilation)\n  $(LSSTTEX): download $(LSSTTEX), providing (among other things) LSST project bib files (not done automatically)\n Options for style: $(styleopts) (Anything else results in generic latex)\n Options for localpip: T or F. Set to T to install mkauthlist in this directory rather than systemwide"


# Interpret `make` with no target as `make tex` (a latex Note).
# At present, if the default_format is anything other than
# [apj|apjl|mnras|prd|prl|emulateapj], a latex Note is made.
# In future, we could think of using `make` to eg run the ipynb
# notebook and make PDF from the output, but this has not been
# implemented yet.
$(main): export flag = \def\flag{${style}}
envflag = \\def\\flag{${style}}
all: $(main) copy

copy:
	cp ${main}.pdf ${outname}.pdf

touch:
	touch ${main}.tex

# if we are in a git repo, add as a submodule; otherwise clone
$(DESCTEX):
	if [ -d .git ]; then git submodule add $(DESCTEXORIGIN); else git clone $(DESCTEXORIGIN); fi
$(LSSTTEX):
	if [ -d .git ]; then git submodule add $(LSSTTEXORIGIN); else git clone $(LSSTTEXORIGIN); fi

#http://journals.aas.org/authors/aastex/linux.html
#change the compiler call to allow a "." file
# {% raw %}
$(main) : $(DESCTEX) authlist
	STYLEFLAG=$(envflag) latexmk $(main)
#	latexmk -g -pdf  \
	-pdflatex='openout_any=a pdflatex %O -interaction=nonstopmode "${flag}\input{%S}"'  \
	${main}
# {% endraw %}

diff: $(DIFF).pdf
$(DIFF).pdf: $(DIFF).tex
	STYLEFLAG=$(envflag) latexmk $<
$(DIFF).tex: $(DRAFT).tex $(main).tex
	latexdiff --exclude-textcmd="multicolumn" $^ > $@
# this ensures that the version to latexdiff against is always updated
.PHONY: $(DRAFT).tex
$(DRAFT).tex:
	if git show $(BASEBRANCH):$(main).tex > $@; then true; else echo "\n\n'make diff' is intended to be used inside a git repository\n"; false; fi



tar : $(main)
	mkdir -p ${tardir}
	cp ${tarfiles} ${tardir} | true
	cp ${outname}.pdf ${tardir}/${outname}.pdf
	cd ${tardir} && tar -czf ../${outname}.tar.gz . && cd ..
	rm -rf ${tardir}

authlist: authors.tex
authors.tex : authors.csv
	$(GETMKAUTH)
	$(MKAUTH) -j ${mastyle} -f -c "LSST Dark Energy Science Collaboration" \
		--cntrb contributions.tex authors.csv authors.tex
#	pip install --upgrade mkauthlist


# http://stackoverflow.com/q/8028314/
.PHONY: $(styleopts)
$(styleopts): export style = $(@)
$(styleopts): export flag = \def\flag{$(@)}
$(styleopts):
	$(MAKE) -e $(main)
	$(MAKE) -e copy
	$(MAKE) -e tar
# NB. the 'tex' target doesn't actually do anything in docswitch - make
# with no target compiles PDF out of main.tex using lsstdescnote.cls
# (which is to say, by default we assume you are writing an LSST
# DESC Note in latex format).

tidy:
	rm -f *.log *.aux *.out *.dvi *.synctex.gz *.fdb_latexmk *.fls
	rm -f *.bbl *.blg *Notes.bib ${main}.pdf

clean: tidy
	rm -f ${outname}.pdf ${outname}.tar.gz

# Update the tex styles etc:

baseurl=https://raw.githubusercontent.com/LSSTDESC/start_paper/deploy

# UPDATES=\
# texmf/bib/apj.bst \
# texmf/bib/mnras.bst \
# texmf/styles/aas_macros.sty \
# texmf/styles/aastex.cls \
# texmf/styles/aastex61.cls \
# texmf/styles/aps_macros.sty \
# texmf/styles/docswitch.sty \
# texmf/styles/emulateapj.cls \
# texmf/styles/mnras.cls \
# texmf/styles/lsstdescnote.cls \
# texmf/styles/lsstdesc_macros.sty \
# texmf/logos/desc-logo-small.png \
# texmf/logos/desc-logo.png \
# texmf/logos/header.png \
# lsstdesc.bib \
# .travis.yml \
# figures/example.png

# .PHONY: $(UPDATES)
# $(UPDATES):
# 	curl -s -S -o $(@) ${baseurl}/$(@)
# 	@echo " "

update: templates
	@echo Updating desc-tex
	cd $(DESCTEX) && git pull
	@echo
	@echo Updating mkauthlist
	pip install --upgrade $(PIPOPTS) mkauthlist
#@echo "\nOver-writing LaTeX style files with the latest versions: \n"
#@mkdir -p .logos figures texmf/styles texmf/bib
#$(MAKE) $(UPDATES)

# Get fresh copies of the templates etc, for reference:
# It is a bad idea to make these phony targets

TEMPLATES:=\
authors.csv \
main.ipynb \
main.md \
main.rst \
main.tex \
main.bib \
Makefile
#.metadata.json \
#.travis.yml 
#figures/example.png
# #acknowledgments.tex \

ALLTEMPLATES := $(TEMPLATES) header.png

# .PHONY: $(TEMPLATES)
# $(TEMPLATES):
# 	curl -s -S -o templates/$(@) ${baseurl}/$(@)
# 	@echo " "

gettemplates := $(foreach t,$(TEMPLATES),curl -s -S -o templates/$(t) ${baseurl}/$(t); )

# NB: fetching header.png will fail as long as desc-tex is private
templates:
	@echo "\nDownloading the latest versions of the template files, for reference: \n"
	@mkdir -p templates
	$(gettemplates)
	curl -f -s -S -o templates/header.png https://raw.githubusercontent.com/LSSTDESC/desc-tex/master/logos/header.png || true
	@echo
	@echo templates/ listing:
	@ls -a templates/*
	@for f in $(ALLTEMPLATES); do diff -q $$f templates/$$f; true; done
#$(MAKE) $(TEMPLATES)
#	$(MAKE) new

# Get a template copy of the latest Makefile, for reference:
# new:
# 	@echo "\nDownloading the latest version of the Makefile, for reference: \n"
# 	@mkdir -p templates
# 	curl -s -S -o templates/Makefile ${baseurl}/Makefile
# 	@echo " "

# Over-write this Makefile with the latest version:
# Why would we include this when a safe option already exists?
# upgrade:
# 	@echo "\nDownloading the latest version of the Makefile: \n"
# 	curl -s -S -o Makefile ${baseurl}/Makefile
# 	@echo "\nUpgrading version of mkauthlist: \n"
# 	pip install mkauthlist --upgrade --no-deps
# 	@echo "\nNow get the latest styles and templates with\n\n    make update\n    make templates\n"
