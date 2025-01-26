all: viamr.pdf

viamr.pdf: viamr.tex viamr.aux viamr.bbl
	pdflatex -shell-escape viamr
	pdflatex -shell-escape viamr

viamr.aux: viamr.tex
	pdflatex -shell-escape viamr
	bibtex viamr

viamr.bbl: viamr.aux viamr.bib
	bibtex viamr

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.lo? *.blg *.snm *.toc *.nav *.vrb *.bbl *.synctex.gz *.thm *.dvi
	@rm -rf result* _minted*
