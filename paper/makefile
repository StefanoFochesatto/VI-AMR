all: viamr.pdf

_figures := drmax herrinf uerrh1 transect
figures := $(addprefix genfigs/, $(addsuffix .png, $(_figures)))

viamr.pdf: viamr.log viamr.tex viamr.bib $(figures)
	while ! [ -f viamr.log ] || grep 'Rerun to get ' viamr.log ; do pdflatex -shell-escape viamr ; bibtex viamr ; done

viamr.log: viamr.tex viamr.bib
	rm -f viamr.log

genfigs/drmax.png genfigs/herrinf.png genfigs/uerrh1.png: genfigs/glacierconv.py
	(cd genfigs/ && python3 glacierconv.py && mogrify -trim *.png)

genfigs/transect.png: genfigs/transect.py
	(cd genfigs/ && python3 transect.py && mogrify -trim transect.png)

convergence/lshapeddomain/h1.png convergence/lshapeddomain/l2.png convergence/lshapeddomain/jaccard.png: convergence/lshapeddomain/generateplots.py
	(cd convergence/lshapeddomain/ && python3 generateplots.py)

convergence/spiral/h1.png convergence/spiral/l2.png convergence/spiral/jaccard.png: convergence/spiral/generateplots.py
	(cd convergence/spiral/ && python3 generateplots.py)

convergence/sphere/h1.png convergence/sphere/l2.png convergence/sphere/jaccard.png: convergence/sphere/generateplots.py
	(cd convergence/sphere/ && python3 generateplots.py)

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.lo? *.blg *.snm *.toc *.nav *.vrb *.bbl *.synctex.gz *.thm *.dvi *.fls *.fdb_latexmk
	@rm -rf result* _minted*
	@rm -f genfigs/*.png
