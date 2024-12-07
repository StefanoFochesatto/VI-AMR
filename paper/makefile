all: viamr.pdf

viamr.pdf: viamr.tex viamr.aux viamr.bbl
	pdflatex viamr
	pdflatex viamr

viamr.aux: viamr.tex
	pdflatex viamr
	bibtex viamr

viamr.bbl: viamr.aux viamr.bib
	bibtex viamr

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.blg *.snm *.toc *.nav *.vrb *.bbl *.synctex.gz *.thm *.dvi
