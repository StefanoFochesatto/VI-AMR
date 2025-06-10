all: viamr.pdf

_figures := drmax herrinf uerrh1 transect
figures := $(addprefix genfigs/, $(addsuffix .png, $(_figures)))

viamr.pdf: viamr.tex viamr.bib $(figures)
	while ! [ -f viamr.log ] || grep 'Rerun to get ' viamr.log ; do pdflatex -shell-escape viamr ; bibtex viamr ; done

genfigs/drmax.png genfigs/herrinf.png genfigs/uerrh1.png: genfigs/glacierconv.py
	(cd genfigs/ && python3 glacierconv.py && mogrify -trim *.png)

genfigs/transect.png: genfigs/transect.py
	(cd genfigs/ && python3 transect.py && mogrify -trim transect.png)

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.lo? *.blg *.snm *.toc *.nav *.vrb *.bbl *.synctex.gz *.thm *.dvi *.fls *.fdb_latexmk
	@rm -rf result* _minted*
	@rm -f genfigs/*.png
